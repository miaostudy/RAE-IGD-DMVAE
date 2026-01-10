import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import importlib
from tqdm import tqdm
from torchvision.utils import save_image
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from collections import OrderedDict, defaultdict
from PIL import Image
import numpy as np
import gc
import time

# --- Project Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "RAE", "src"))

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from IGD.data import load_data
import IGD.train_models.resnet as RN

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"


# ==========================================
# 1. Latent Proxy Definition
# ==========================================
class LatentProxy(nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased Dropout
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased Dropout
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.net(x)


# ==========================================
# 2. Helper Functions
# ==========================================
@torch.no_grad()
def extract_latents_and_normalize(model, data_loader, device, max_batches=200):
    print(f"Extracting latents (max_batches={max_batches})...")
    model.eval()
    latents_list = []
    labels_list = []

    for i, (x, y) in enumerate(tqdm(data_loader, desc="Extracting")):
        if i >= max_batches: break
        x = x.to(device)
        z = model.encode(x)
        if isinstance(z, tuple): z = z[0]
        if hasattr(z, 'sample'): z = z.mode()

        latents_list.append(z.cpu())
        labels_list.append(y)

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Calculate Stats
    mean = latents.mean()
    std = latents.std()
    print(f"Original Latents Stats -> Mean: {mean:.4f}, Std: {std:.4f}")

    # Normalize
    latents = (latents - mean) / (std + 1e-6)
    print(f"Normalized Latents -> Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")

    # Return stats as tensor on device for later use
    stats = {'mean': mean.to(device).detach(), 'std': std.to(device).detach()}

    return latents, labels, stats


def train_proxy(latents, labels, device, num_classes=1000, epochs=5):  # Epochs 改回 5 或保持 3 均可
    print(f"Training Proxy on {len(labels)} samples (Epochs={epochs}, w/ Noise Augmentation)...")
    input_dim = np.prod(latents.shape[1:])

    proxy = LatentProxy(input_dim, num_classes).to(device)
    # 保持 weight_decay 防止 Logits 爆炸
    optimizer = optim.Adam(proxy.parameters(), lr=1e-3, weight_decay=1e-4)
    # 保持 label_smoothing 防止梯度消失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    dataset = TensorDataset(latents, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    proxy.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for z_batch, y_batch in loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)

            # === [关键修改]：加入高斯噪声 ===
            # std=1.0 是因为我们之前把 latents 归一化到了 std=1
            # 0.1~0.2 的噪声强度通常比较合适，模拟估计误差
            noise = torch.randn_like(z_batch) * 0.15
            z_noisy = z_batch + noise
            # ==============================

            optimizer.zero_grad()
            output = proxy(z_noisy)  # 使用加噪后的输入
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

        print(f"  Proxy Epoch {epoch + 1}/{epochs} | Acc: {correct / total:.4f} | Loss: {total_loss / len(loader):.4f}")

    proxy.eval()
    return proxy


def get_real_gradients(proxy, latents, labels, target_class_idx, device, grad_ipc=80):
    """
    Calculate average gradient of proxy parameters on REAL (Normalized) data.
    """
    proxy.eval()
    # Ensure criterion matches training (label smoothing) to keep gradients consistent
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    cls_mask = (labels == target_class_idx)
    cls_latents = latents[cls_mask]

    # print(f"[Debug] get_real_gradients for Class {target_class_idx}: Found {len(cls_latents)} samples.")

    if len(cls_latents) == 0:
        return [torch.zeros_like(p) for p in proxy.parameters()]

    if len(cls_latents) > grad_ipc:
        cls_latents = cls_latents[:grad_ipc]

    cls_latents = cls_latents.to(device).requires_grad_(True)
    targets = torch.full((len(cls_latents),), target_class_idx, dtype=torch.long, device=device)

    proxy.zero_grad()
    logits = proxy(cls_latents)
    loss = criterion(logits, targets)
    grads = torch.autograd.grad(loss, proxy.parameters())

    # Debug: Check gradient norm
    # total_norm = sum([g.norm().item() for g in grads])
    # print(f"[Debug] Real Gradient Total Norm: {total_norm:.6f}")

    return [g.detach() for g in grads]


# ==========================================
# 3. Config Helpers
# ==========================================
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class DecoderWrapper:
    def __init__(self, decode_fn):
        self.decode_fn = decode_fn

    def decode(self, z): return self.decode_fn(z)


# ==========================================
# 4. Sampling Logic (IGD with Normalization Fix)
# ==========================================
# ==========================================
# Fixed Sampling Logic (Graph Connectivity)
# ==========================================
@torch.no_grad()
def igd_ode_sample(model, z, steps, model_kwargs, device):
    x = z
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    # --- Resources ---
    latent_proxy = model_kwargs.get('latent_proxy', None)
    real_grads = model_kwargs.get('real_grads', None)
    memory_bank = model_kwargs.get('memory_bank', [])

    # Stats for normalization
    stats = model_kwargs.get('stats', {'mean': 0.0, 'std': 1.0})
    mean_val = stats['mean']
    std_val = stats['std']

    # --- Parameters ---
    div_scale = model_kwargs.get('gamma', 120.0)  # Diversity
    inf_scale = model_kwargs.get('k', 5.0)  # Influence

    low, high = model_kwargs.get('low', 0), model_kwargs.get('high', 1000)
    target_y_global = model_kwargs['y']
    target_y_proxy = model_kwargs.get('y_proxy', target_y_global)

    is_cfg = (x.shape[0] == 2 * target_y_proxy.shape[0])

    # Match training criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Prep Memory (Normalized)
    mem_tensor = None
    if len(memory_bank) > 0 and div_scale > 0:
        mem_tensor = torch.cat(memory_bank, dim=0).to(device)
        mem_flat = mem_tensor.flatten(start_dim=1)
        # Detached memory doesn't need grad
        mem_norm = mem_flat / (mem_flat.norm(dim=1, keepdim=True) + 1e-8)

    for i in tqdm(range(steps), desc="ODE Sampling"):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr  # Negative

        t_in = torch.full((x.shape[0],), t_curr, device=device, dtype=torch.float)

        # 1. Forward DiT
        # We need gradients w.r.t input x_in_raw later, so v_pred used in guidance must be consistent
        if 'cfg_scale' in model_kwargs and model_kwargs['cfg_scale'] > 1.0:
            v_pred = model(x, t_in, model_kwargs['y'])
        else:
            v_pred = model(x, t_in, model_kwargs['y'])

        # 2. IGD Guidance
        t_check = t_curr.item() * 1000
        should_guide = (t_check >= low) and (t_check <= high) and (latent_proxy is not None)

        inf_loss_val = 0.0
        div_loss_val = 0.0

        if should_guide:
            x_0_est = x - t_curr * v_pred

            if is_cfg:
                x_0_cond, _ = x_0_est.chunk(2)
                v_cond, _ = v_pred.chunk(2)
            else:
                x_0_cond = x_0_est
                v_cond = v_pred

            # Detach and require grad for guidance calculation
            x_in_raw = x_0_cond.detach().requires_grad_(True)

            total_guidance = torch.zeros_like(x_in_raw)

            # 【CRITICAL FIX】: 全部计算必须在 enable_grad 内部
            with torch.enable_grad():
                # Normalize inputs for Proxy
                x_in_norm = (x_in_raw - mean_val) / (std_val + 1e-6)

                # --- A. Influence Guidance (Gradient Matching) ---
                if real_grads is not None and inf_scale > 0:
                    logits = latent_proxy(x_in_norm)
                    loss_gen = criterion(logits, target_y_proxy)

                    # 1. Gradients w.r.t parameters
                    gen_grads = torch.autograd.grad(loss_gen, latent_proxy.parameters(), create_graph=True)

                    # 2. Match Loss
                    match_loss = 0
                    for g_gen, g_real in zip(gen_grads, real_grads):
                        match_loss += ((g_gen - g_real) ** 2).sum()

                    inf_loss_val = match_loss.item()

                    should_retain = (mem_tensor is not None and div_scale > 0)
                    # 3. Gradients w.r.t input x_in_raw
                    grad_inf_tuple = torch.autograd.grad(match_loss, x_in_raw, allow_unused=True,
                                                         retain_graph=should_retain)

                    if grad_inf_tuple[0] is not None:
                        grad_inf = grad_inf_tuple[0]

                        # Adaptive Scaling & Application
                        v_norm = (v_cond.detach() ** 2).mean().sqrt()
                        g_inf_norm = (grad_inf ** 2).mean().sqrt() + 1e-8

                        scaled_grad_inf = grad_inf * (v_norm / g_inf_norm) * inf_scale
                        scaled_grad_inf = torch.clamp(scaled_grad_inf, -0.5, 0.5)

                        total_guidance = total_guidance + scaled_grad_inf

                # --- B. Deviation Guidance (Diversity) ---
                if mem_tensor is not None and div_scale > 0:
                    x_flat = x_in_norm.flatten(start_dim=1)
                    x_vec = x_flat / (x_flat.norm(dim=1, keepdim=True) + 1e-8)

                    sims = torch.mm(x_vec, mem_norm.t())
                    if sims.numel() > 0:
                        max_sim, _ = sims.max(dim=1)
                        loss_div = max_sim.sum()
                        div_loss_val = loss_div.item()

                        grad_div_tuple = torch.autograd.grad(loss_div, x_in_raw, allow_unused=True)
                        if grad_div_tuple[0] is not None:
                            grad_div = grad_div_tuple[0]

                            v_norm = (v_cond.detach() ** 2).mean().sqrt()
                            g_div_norm = (grad_div ** 2).mean().sqrt() + 1e-8

                            scaled_grad_div = grad_div * (v_norm / g_div_norm) * div_scale
                            scaled_grad_div = torch.clamp(scaled_grad_div, -0.5, 0.5)

                            total_guidance = total_guidance + scaled_grad_div

            total_guidance = total_guidance.detach()
            # --- Apply Total Guidance ---
            if is_cfg:
                v_cond_new = v_cond + total_guidance
                v_uncond = v_pred.chunk(2)[1]
                v_pred = torch.cat([v_cond_new, v_uncond], dim=0)
            else:
                v_pred = v_pred + total_guidance

        if should_guide and i % 10 == 0:
            print(f"  Step {i} | t={t_curr:.3f} | L_Inf: {inf_loss_val:.4f} | L_Div: {div_loss_val:.4f}")

        x = x + v_pred * dt

    return x


# ==========================================
# 5. Main
# ==========================================
def main(args):
    torch.manual_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Config Loading ---
    config = None
    if args.config and os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if 'stage_2' in config and 'target' in config['stage_2']:
            target_str = config['stage_2']['target']
            if 'stage2.dmvae_models' in target_str:
                config['stage_2']['target'] = target_str.replace('stage2.dmvae_models', 'stage2.models')

    # --- Class Selection ---
    if args.spec == 'cifar10':
        sel_classes = [str(i) for i in range(10)]
        class_labels = [int(x) for x in sel_classes]
    else:
        if args.spec == 'woof':
            list_file = 'IGD/misc/class_woof.txt'
        elif args.spec == 'nette':
            list_file = 'IGD/misc/class_nette.txt'
        elif args.spec == '1k':
            list_file = 'IGD/misc/class_indices.txt'
        else:
            list_file = 'IGD/misc/class100.txt'

        if os.path.exists(list_file):
            with open(list_file, 'r') as fp:
                sel_classes_raw = [s.strip() for s in fp.readlines()]
        else:
            sel_classes_raw = []

        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = args.nclass * (phase + 1)
        if len(sel_classes_raw) > 0:
            sel_classes = sel_classes_raw[cls_from:cls_to]
            master_file = 'IGD/misc/class_indices.txt'
            if os.path.exists(master_file):
                with open(master_file, 'r') as f:
                    all_classes = [l.strip() for l in f.readlines()]
                class_labels = [all_classes.index(c) for c in sel_classes if c in all_classes]
            else:
                class_labels = list(range(cls_from, cls_to))
        else:
            sel_classes = []
            class_labels = []

    # --- Model Init ---
    if args.model_type == 'rae' and config:
        print("Initializing RAE from config...")
        rae = instantiate_from_config(config['stage_1']).to(device).eval()
        if args.rae_ckpt:
            print(f"Loading RAE checkpoint: {args.rae_ckpt}")
            st = torch.load(args.rae_ckpt, map_location='cpu')
            if 'model' in st: st = st['model']
            rae.load_state_dict(st, strict=False)

        model = instantiate_from_config(config['stage_2']).to(device).eval()
        dit_ckpt_path = args.dit_ckpt if args.dit_ckpt else config['stage_2'].get('ckpt')
        if dit_ckpt_path:
            print(f"Loading DiT checkpoint: {dit_ckpt_path}")
            ckpt = torch.load(dit_ckpt_path, map_location='cpu')
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            elif 'ema' in ckpt:
                model.load_state_dict(ckpt['ema'])
            else:
                model.load_state_dict(ckpt)

        def decode_fn(z):
            img = rae.decode(z)
            return img

        if 'misc' in config and 'latent_size' in config['misc']:
            c, h, w = config['misc']['latent_size']
            latent_size = h * 16
        else:
            latent_size = args.image_size // 16
    else:
        raise NotImplementedError

    # --- Proxy Setup ---
    print("\n[Proxy Setup] Loading training data...")
    latent_proxy = None
    proxy_class_map = {}
    all_latents_tensor = None
    all_labels_tensor = None
    latent_stats = None

    try:
        dataset_obj, train_loader, _, num_classes_data = load_data(args, tsne=False)

        if hasattr(dataset_obj, 'class_to_idx'):
            proxy_class_map = dataset_obj.class_to_idx
        elif hasattr(train_loader.dataset, 'class_to_idx'):
            proxy_class_map = train_loader.dataset.class_to_idx
        elif hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'class_to_idx'):
            proxy_class_map = train_loader.dataset.dataset.class_to_idx
        else:
            for idx, name in enumerate(sorted(sel_classes)):
                proxy_class_map[name] = idx

        # Extract & Normalize & Get Stats
        latents, labels, stats = extract_latents_and_normalize(rae, train_loader, device, max_batches=500)
        all_latents_tensor = latents
        all_labels_tensor = labels
        latent_stats = stats

        # =========================================================
        # FIX 1: Remap ImageNette labels (0-9) to ImageNet original IDs
        # =========================================================
        if args.spec == 'nette' and len(class_labels) == 10:
            print("[Info] Remapping ImageNette dataset labels (0-9) to ImageNet original IDs...")
            mapper = torch.tensor(class_labels, dtype=torch.long, device=labels.device)
            labels = mapper[labels.long()]
            num_classes_data = 1000
            all_labels_tensor = labels
        # =========================================================

        # Train with reduced epochs
        latent_proxy = train_proxy(latents, labels, device, num_classes=num_classes_data, epochs=3)

    except Exception as e:
        print(f"Warning: Failed to train proxy. Error: {e}")
        import traceback
        traceback.print_exc()

    # --- Sampling Loop ---
    print(f"\n[Sampling] Start generating {args.num_samples} samples per class...")
    os.makedirs(args.save_dir, exist_ok=True)
    batch_size = 1

    class_memory_bank = defaultdict(list)

    for i, class_label in enumerate(class_labels):
        sel_class_name = sel_classes[i] if i < len(sel_classes) else str(class_label)

        # =========================================================
        # FIX 2: Ensure Proxy Label ID matches the Remapped ID
        # =========================================================
        if args.spec == 'nette':
            proxy_label_idx = class_label
        elif sel_class_name in proxy_class_map:
            proxy_label_idx = proxy_class_map[sel_class_name]
        else:
            proxy_label_idx = class_label
        # =========================================================

        save_class_dir = os.path.join(args.save_dir, sel_class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        print(f"Generating class: {sel_class_name} (DiT:{class_label}, Proxy:{proxy_label_idx})")

        # Get Real Gradients (Influence Target)
        current_real_grads = None
        if latent_proxy is not None and all_latents_tensor is not None:
            current_real_grads = get_real_gradients(latent_proxy, all_latents_tensor, all_labels_tensor,
                                                    proxy_label_idx, device, grad_ipc=80)

        for shift in tqdm(range(0, args.num_samples, batch_size)):
            if config and 'misc' in config:
                C, H, W = config['misc']['latent_size']
            else:
                C, H, W = 32, latent_size, latent_size

            z = torch.randn(batch_size, C, H, W, device=device)
            y = torch.tensor([class_label] * batch_size, device=device)
            y_p = torch.tensor([proxy_label_idx] * batch_size, device=device)

            if args.cfg_scale > 1.0:
                z_in = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y_in = torch.cat([y, y_null], 0)
            else:
                z_in = z
                y_in = y

            current_mem = class_memory_bank[class_label]

            model_kwargs = dict(
                y=y_in,
                cfg_scale=args.cfg_scale,
                latent_proxy=latent_proxy,
                real_grads=current_real_grads,
                gamma=args.gamma,
                k=args.k,
                memory_bank=current_mem,
                stats=latent_stats,
                low=args.low,
                high=args.high,
                y_proxy=y_p
            )

            samples = igd_ode_sample(model, z_in, args.num_sampling_steps, model_kwargs, device)

            if args.cfg_scale > 1.0:
                valid_sample, _ = samples.chunk(2, dim=0)
            else:
                valid_sample = samples

            mean_v = latent_stats['mean']
            std_v = latent_stats['std']
            valid_sample_norm = (valid_sample.detach() - mean_v) / (std_v + 1e-6)
            class_memory_bank[class_label].append(valid_sample_norm)

            imgs = decode_fn(valid_sample)

            for j, img in enumerate(imgs):
                idx = shift + j + args.total_shift
                save_path = os.path.join(save_class_dir, f"{idx}.png")
                save_image(img, save_path, normalize=True, value_range=(-1, 1))

        del class_memory_bank[class_label]
        gc.collect()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_type", type=str, default='rae')

    parser.add_argument("--rae_ckpt", type=str, default=None)
    parser.add_argument("--dit_ckpt", type=str, default=None)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default='./results/generated')
    parser.add_argument("--total-shift", type=int, default=0)

    # IGD Parameters
    parser.add_argument("--gamma", type=float, default=120.0, help="IGD Diversity Weight (Deviation)")
    parser.add_argument("--k", type=float, default=5.0, help="IGD Influence Weight (Gradient Matching)")

    parser.add_argument("--low", type=int, default=0)
    parser.add_argument("--high", type=int, default=1000)

    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--data_dir", type=str, default='/data')
    parser.add_argument('--imagenet_dir', nargs='+', default=['/data/imagenet'])
    parser.add_argument("--nclass", type=int, default=1000)
    parser.add_argument("--spec", type=str, default='nette')
    parser.add_argument("--phase", type=int, default=0)

    parser.add_argument("--augment", action='store_true', default=False)
    parser.add_argument("--slct_type", type=str, default='random')
    parser.add_argument("--ipc", type=int, default=-1)
    parser.add_argument("--load_memory", action='store_true', default=False)
    parser.add_argument("--dseed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_real", type=int, default=32)
    parser.add_argument("--device", type=str, default='cuda')

    args = parser.parse_args()
    main(args)