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

# 引入 create_diffusion
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from IGD.data import load_data
import IGD.train_models.resnet as RN

# 引入 ReparamModule (确保路径正确)
try:
    from IGD.reparam_module import ReparamModule
except ImportError:
    from reparam_module import ReparamModule

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================
# 1. Latent Proxy Definition
# ==========================================
class LatentProxy(nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
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


def train_proxy(latents, labels, device, num_classes=1000, epochs=5):
    print(f"Training Proxy on {len(labels)} samples...")
    input_dim = np.prod(latents.shape[1:])

    proxy = LatentProxy(input_dim, num_classes).to(device)
    optimizer = optim.Adam(proxy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(latents, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    proxy.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for z_batch, y_batch in loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = proxy(z_batch)
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


def get_real_gradients(proxy, latents, labels, target_class_idx, device, grad_ipc=200):
    """
    Calculate average gradient of proxy parameters on REAL (Normalized) data.
    """
    proxy.eval()
    proxy.zero_grad()  # Clear any existing grads
    criterion = nn.CrossEntropyLoss()

    cls_mask = (labels == target_class_idx)
    cls_latents = latents[cls_mask]

    if len(cls_latents) == 0:
        return [torch.zeros_like(p) for p in proxy.parameters()]

    if len(cls_latents) > grad_ipc:
        cls_latents = cls_latents[:grad_ipc]

    cls_latents = cls_latents.to(device).requires_grad_(True)
    targets = torch.full((len(cls_latents),), target_class_idx, dtype=torch.long, device=device)

    logits = proxy(cls_latents)
    loss = criterion(logits, targets)

    # proxy here is ReparamModule, parameters() returns [flat_param]
    grads = torch.autograd.grad(loss, proxy.parameters())

    # Return list of gradients (usually just one flat tensor)
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


# ==========================================
# 4. Main
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
        # Simplified class loading
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
            return torch.clamp(img, -1, 1)

        if 'misc' in config and 'latent_size' in config['misc']:
            c, h, w = config['misc']['latent_size']
            latent_size = h * 16
        else:
            latent_size = args.image_size // 16
    else:
        raise NotImplementedError

    # --- Diffusion Setup ---
    # [FIX] Set learn_sigma=False because LightningDiT discards variance in forward()
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=False)

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
        latents, labels, stats = extract_latents_and_normalize(rae, train_loader, device, max_batches=100)
        all_latents_tensor = latents
        all_labels_tensor = labels
        latent_stats = stats

        # Train Proxy
        latent_proxy = train_proxy(latents, labels, device, num_classes=num_classes_data, epochs=5)

    except Exception as e:
        print(f"Warning: Failed to train proxy. Error: {e}")
        import traceback
        traceback.print_exc()

    # --- Prepare Proxy for Reparam ---
    surrogate = None
    flat_proxy_params = None

    if latent_proxy is not None:
        surrogate = ReparamModule(latent_proxy)
        surrogate.eval()
        # [FIX] Access flat_param directly from ReparamModule
        flat_proxy_params = surrogate.flat_param.detach().clone()

    # --- Sampling Loop ---
    print(f"\n[Sampling] Start generating {args.num_samples} samples per class using Diffusion Loop...")
    os.makedirs(args.save_dir, exist_ok=True)
    batch_size = 1

    criterion_ce = nn.CrossEntropyLoss().to(device)
    class_memory_bank = defaultdict(list)

    for i, class_label in enumerate(class_labels):
        sel_class_name = sel_classes[i] if i < len(sel_classes) else str(class_label)

        if sel_class_name in proxy_class_map:
            proxy_label_idx = proxy_class_map[sel_class_name]
        else:
            proxy_label_idx = class_label
            if latent_proxy is not None and proxy_label_idx >= 10:
                proxy_label_idx = 0

        save_class_dir = os.path.join(args.save_dir, sel_class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        print(f"Generating class: {sel_class_name} (DiT:{class_label}, Proxy:{proxy_label_idx})")

        # 1. Calculate Real Gradients
        real_grads_list = []
        if latent_proxy is not None and all_latents_tensor is not None:
            # [FIX] Use args.gi
            grads_per_layer = get_real_gradients(surrogate, all_latents_tensor, all_labels_tensor,
                                                 proxy_label_idx, device, grad_ipc=args.gi)
            if len(grads_per_layer) > 0:
                flat_real_grad = grads_per_layer[0]
                real_grads_list = [flat_real_grad]

        ckpts_list = [flat_proxy_params] if flat_proxy_params is not None else []

        for shift in tqdm(range(0, args.num_samples, batch_size)):
            if config and 'misc' in config:
                C, H, W = config['misc']['latent_size']
            else:
                C, H, W = 32, latent_size, latent_size

            # Create Noise
            z = torch.randn(batch_size, C, H, W, device=device)
            y = torch.tensor([class_label] * batch_size, device=device)

            if args.cfg_scale > 1.0:
                z_in = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y_in = torch.cat([y, y_null], 0)
            else:
                z_in = z
                y_in = y

            current_mem = class_memory_bank[class_label]

            # IGD Parameters
            gm_resource = [
                rae,
                surrogate,
                ckpts_list,
                real_grads_list,
                proxy_label_idx,
                criterion_ce,
                1,
                1,
                args.k
            ]

            model_kwargs = dict(
                y=y_in,
                cfg_scale=args.cfg_scale,
                gen_type='igd_latent',
                low=args.low,
                high=args.high,
                pseudo_memory_c=current_mem,
                neg_e=args.gamma,
                latent_stats=latent_stats,
                gm_resource=gm_resource
            )

            # [FIX] Wrapper to filter kwargs for DiT
            def model_fn_wrapper(x, t, **kwargs):
                model_args = {}
                if 'y' in kwargs:
                    model_args['y'] = kwargs['y']
                if 'cfg_scale' in kwargs:
                    model_args['cfg_scale'] = kwargs['cfg_scale']
                return model.forward_with_cfg(x, t, **model_args)

            # [FIX] Use keyword arguments for p_sample_loop
            samples_dict = diffusion.p_sample_loop(
                model=model_fn_wrapper,
                shape=z_in.shape,
                noise=z_in,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device
            )

            final_sample = samples_dict['sample'] if isinstance(samples_dict, dict) else samples_dict

            if args.cfg_scale > 1.0:
                valid_sample, _ = final_sample.chunk(2, dim=0)
            else:
                valid_sample = final_sample

            # Save to memory bank (normalized)
            mean_v = latent_stats['mean']
            std_v = latent_stats['std']
            valid_sample_norm = (valid_sample.detach() - mean_v) / (std_v + 1e-6)
            class_memory_bank[class_label].append(valid_sample_norm)

            # Decode and Save
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

    # [NEW] Added --gi parameter
    parser.add_argument("--gi", type=int, default=200, help="Gradient IPC (Number of real samples for influence calc)")

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