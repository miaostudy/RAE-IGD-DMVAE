import os
import sys
import torch
import yaml
import importlib
from tqdm import tqdm
import torch.distributed as dist

# 优化 CUDA 设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from data import ImageFolder, ImageFolder_mp, CIFAR10_mp
from collections import OrderedDict, defaultdict
from PIL import Image
import numpy as np
import gc
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
import time
from reparam_module import ReparamModule
import torch.nn as nn

# --- Path Setup & Imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "RAE", "src"))

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# --- Helper Functions ---


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


def define_model(args, nclass, logger=None, size=None):
    args.size = 256
    args.width = 1.0
    args.norm_type = 'instance'
    args.nch = 3
    if size == None: size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.spec, args.depth, nclass, norm_type=args.norm_type, size=size, nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.spec, args.depth, nclass, width=args.width, norm_type=args.norm_type, size=size,
                              nch=args.nch)
    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        model = CN.ConvNet(channel=4, num_classes=nclass, net_width=128, net_depth=3, net_act='relu',
                           net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
        model.classifier = nn.Linear(2048, nclass)
    elif args.net_type == 'convnet6':
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=6, net_act='relu',
                           net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
    elif args.net_type == 'convnet4':
        model = CN.ConvNet(channel=args.nch, num_classes=nclass, net_width=128, net_depth=4, net_act='relu',
                           net_norm='instancenorm', net_pooling='avgpooling', im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))
    return model


def rand_ckpts(args):
    if args.net_type == 'convnet6':
        expert_path = './%s/%s/%s/' % (args.ckpt_path, args.spec, args.net_type)
    else:
        expert_path = './%s/%s/%s%s/' % (args.ckpt_path, args.spec, args.net_type, args.depth)

    expert_files = os.listdir(expert_path)
    rand_id1 = 0
    state = torch.load(os.path.join(expert_path, expert_files[rand_id1]))
    # print('file name:', expert_path + expert_files[rand_id1])
    ckpts = state[0]

    if args.spec == 'woof':
        if args.ckpt_path.startswith('ckpts'):
            if args.net_type == 'convnet6':
                idxs = [0, 5, 16, 40]
            elif args.net_type == 'resnet_ap':
                idxs = [0, 6, 16, 39]
            elif args.net_type == 'resnet':
                idxs = [0, 16, 33]
        elif args.ckpt_path.startswith('cut_ckpts'):
            if args.net_type == 'convnet4':
                idxs = [1, 4, 13, 27, 57]
            elif args.net_type == 'convnet6':
                idxs = [0, 10, 26, 60]
    elif args.spec == 'nette':
        if args.ckpt_path.startswith('ckpt'):
            if args.net_type == 'convnet6':
                idxs = [0, 2, 11, 40]
            elif args.net_type == 'resnet_ap':
                idxs = [0, 6, 16, 39]
            elif args.net_type == 'resnet':
                idxs = [0, 8, 27]
    elif args.spec == '1k':
        if args.ckpt_path.startswith('ckpt'):
            if args.net_type == 'convnet6': idxs = [0, 5, 18, 50]
    elif args.spec == 'cifar10':
        if args.ckpt_path.startswith('ckpt'):
            if args.net_type == 'convnet6':
                idxs = [0, 5, 18, 50]
            elif args.net_type == 'resnet_ap':
                idxs = [0, 6, 16, 39]
            elif args.net_type == 'resnet':
                idxs = [0, 8, 27]

    # print('ckpt idxs:',idxs)
    return [ckpts[ii] for ii in idxs]


def get_grads(sel_classes, class_labels, sel_class, ckpts, surrogate, device):
    criterion_ce = nn.CrossEntropyLoss().to(device)
    correspond_labels = defaultdict(list)
    grads_memory = []
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    if args.spec == 'cifar10':
        dataset = CIFAR10_mp(args.data_path, train=True, transform=transform,
                             download=True, sel_class=sel_class, return_origin=True)
    else:
        dataset = ImageFolder_mp(args.data_path, transform=transform, nclass=args.nclass,
                                 ipc=args.real_ipc, spec=args.spec, phase=args.phase,
                                 seed=0, return_origin=True, sel_class=sel_class)

    real_loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    for x, ry, y in real_loader:
        x = x.to(device)
        y_val = int(y.numpy()[0])

        grads_memory.extend(x.detach().split(1))
        correspond_labels[y_val] = ry[0].cpu().numpy()

        if len(grads_memory) > args.grad_ipc:
            break

    grads_memory = grads_memory[:args.grad_ipc]
    y_target = y_val

    real_gradients = defaultdict(list)
    gap = 50
    gap_idxs = np.arange(0, args.grad_ipc, gap).tolist()
    correspond_y = correspond_labels[y_target]
    ckpt_grads = []


    for ii, ckpt in enumerate(ckpts):
        for gi in gap_idxs:
            cur_embd0 = torch.stack(grads_memory[gi:gi + gap]).cpu().numpy()
            cur_embds = torch.from_numpy(cur_embd0).squeeze(1).to(device).requires_grad_(True)
            cur_params = torch.cat([p.data.to(device).reshape(-1) for p in ckpt], 0).requires_grad_(True)

            if gi == 0:
                acc_grad = torch.zeros(cur_params.shape)

            real_pred = surrogate(cur_embds, flat_param=cur_params)
            real_target = torch.tensor([np.ones(len(cur_embds)) * correspond_y], dtype=torch.long,
                                       device=device).view(-1)
            real_loss = criterion_ce(real_pred, real_target)

            real_grad = torch.autograd.grad(real_loss, cur_params)[0]
            acc_grad += real_grad.detach().data.cpu()

        ckpt_grads.append(acc_grad / len(gap_idxs))

    real_gradients[y_target] = ckpt_grads

    return real_gradients, y_target, correspond_labels


def gm_loss(pg, rg):
    inv_gm_dist = torch.sum(
        1 - torch.sum(pg * rg, dim=-1) / (torch.norm(pg, dim=-1) * torch.norm(rg, dim=-1) + 0.000001))
    return inv_gm_dist


def igd_ode_sample(model, z, steps, model_kwargs, device):
    """
    Euler ODE sampler with IGD gradient guidance + AMP.
    """
    x = z
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    decoder, surrogate, ckpts, real_grad, label_idx, criterion, repeat, repeat_init, gm_scale = model_kwargs[
        'gm_resource']
    neg_e = model_kwargs.get('neg_e', 0.0)
    low, high = model_kwargs.get('low', 0), model_kwargs.get('high', 1000)

    for i in tqdm(range(steps), desc="ODE Sampling", disable=dist.get_rank() != 0):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_in = torch.full((x.shape[0],), t_curr, device=device, dtype=torch.float)

        # 1. Forward model to get Velocity (Use AMP)
        with torch.cuda.amp.autocast():
            if 'cfg_scale' in model_kwargs and model_kwargs['cfg_scale'] > 1.0:
                v_pred = model.forward_with_cfg(x, t_in, model_kwargs['y'], model_kwargs['cfg_scale'])
            else:
                v_pred = model(x, t_in, model_kwargs['y'])

        # 2. IGD Guidance
        t_check = t_curr.item() * 1000
        should_guide = (t_check >= low) and (t_check <= high)
        guidance_grad = torch.zeros_like(x)

        if should_guide and gm_scale > 0:
            with torch.enable_grad():  # Gradient calculation needs enabled grad
                with torch.cuda.amp.autocast():  # Use AMP for heavy gradient graph
                    x_in = x.detach().requires_grad_(True)

                    if 'cfg_scale' in model_kwargs and model_kwargs['cfg_scale'] > 1.0:
                        v_pred_grad = model.forward_with_cfg(x_in, t_in, model_kwargs['y'], model_kwargs['cfg_scale'])
                    else:
                        v_pred_grad = model(x_in, t_in, model_kwargs['y'])

                    # x0 estimation: x0 = xt - t * v
                    x_0_est = x_in - t_curr * v_pred_grad

                    # Decode to image space
                    pseudo_imgs = decoder.decode(x_0_est)

                    # Gradient against Surrogate
                    pseudo_target = torch.tensor([np.ones(len(pseudo_imgs)) * label_idx], dtype=torch.long,
                                                 device=device).view(-1)

                    total_gm_loss = 0
                    for idx, ckpt in enumerate(ckpts):
                        # Move params to GPU only when needed
                        cur_params = torch.cat([p.data.to(device).reshape(-1) for p in ckpt], 0).requires_grad_(True)
                        real_g = real_grad[idx].to(device)

                        pseudo_pred = surrogate(pseudo_imgs, flat_param=cur_params)
                        pseudo_loss_val = criterion(pseudo_pred, pseudo_target)

                        pseudo_g = torch.autograd.grad(pseudo_loss_val, cur_params, create_graph=True)[0]
                        total_gm_loss += gm_loss(pseudo_g, real_g)

                        # Free up
                        del cur_params, pseudo_g

                    total_gm_loss /= len(ckpts)

                # End of autocast, calculate gradient of x_in
                guidance_grad = torch.autograd.grad(total_gm_loss, x_in)[0]

                # Cleanup graph
                del x_in, x_0_est, pseudo_imgs, pseudo_pred, total_gm_loss
                # torch.cuda.empty_cache() # Optional: frequent empty_cache slows down, use only if critical

        if should_guide:
            v_norm = (v_pred.detach() ** 2).mean().sqrt()
            g_norm = (guidance_grad ** 2).mean().sqrt() + 1e-6
            adaptive_scale = (v_norm / g_norm) * gm_scale
            v_pred = v_pred + guidance_grad * adaptive_scale

        x = x + v_pred * dt

    return x


class DecoderWrapper:
    def __init__(self, decode_fn):
        self.decode_fn = decode_fn

    def decode(self, z):
        return self.decode_fn(z)


def main(args):
    # --- Distributed Init ---
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if rank == 0:
            print(f"Distributed Init: World Size {world_size}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running in single process mode.")

    # Seed
    torch.manual_seed(args.seed + rank)  # Seed offset by rank
    np.random.seed(args.seed + rank)

    # --- Config Loading ---
    config = None
    if args.config and os.path.exists(args.config):
        if rank == 0: print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        if 'stage_2' in config and 'target' in config['stage_2']:
            target_str = config['stage_2']['target']
            if 'stage2.dmvae_models' in target_str:
                new_target = target_str.replace('stage2.dmvae_models', 'stage2.models')
                config['stage_2']['target'] = new_target

    # --- Class Selection Logic ---
    if args.spec == 'cifar10':
        sel_classes = [str(i) for i in range(10)]
        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = args.nclass * (phase + 1)
        sel_classes = sel_classes[cls_from:cls_to]
        class_labels = [int(x) for x in sel_classes]
    else:
        with open('IGD/misc/class_indices.txt', 'r') as fp:
            all_classes = [line.strip() for line in fp.readlines()]

        if args.spec == 'woof':
            file_list = 'IGD/misc/class_woof.txt'
        elif args.spec == 'nette':
            file_list = 'IGD/misc/class_nette.txt'
        elif args.spec == '1k':
            file_list = 'IGD/misc/class_indices.txt'
        else:
            file_list = 'IGD/misc/class100.txt'

        with open(file_list, 'r') as fp:
            sel_classes_raw = [s.strip() for s in fp.readlines()]

        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = args.nclass * (phase + 1)
        sel_classes = sel_classes_raw[cls_from:cls_to]
        class_labels = [all_classes.index(c) for c in sel_classes]

    # --- Model Initialization (Load to specific device) ---
    model_type = args.model_type

    if model_type == 'rae' and config:
        if rank == 0: print("Initializing RAE from config...")
        rae = instantiate_from_config(config['stage_1']).to(device).eval()
        if args.rae_ckpt:
            rae.load_state_dict(torch.load(args.rae_ckpt, map_location='cpu'), strict=False)

        model = instantiate_from_config(config['stage_2']).to(device).eval()

        dit_ckpt_path = args.dit_ckpt if args.dit_ckpt else config['stage_2'].get('ckpt')
        if dit_ckpt_path:
            if rank == 0: print(f"Loading DiT checkpoint: {dit_ckpt_path}")
            ckpt = torch.load(dit_ckpt_path, map_location='cpu')
            if 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            elif 'ema' in ckpt:
                model.load_state_dict(ckpt['ema'])
            else:
                model.load_state_dict(ckpt)

        def decode_fn(z):
            return rae.decode(z)

        if 'misc' in config and 'latent_size' in config['misc']:
            c, h, w = config['misc']['latent_size']
            latent_size = h * 16
            latent_size = args.image_size // 16

    elif model_type == 'dmvae':
        from DMVAE.diffusion.lightningdit.lightningdit import LightningDiT as DMVAE_DiT
        from DMVAE.dmvae_models.vae import VAE as DMVAE_VAE

        if rank == 0: print("Initializing DMVAE...")
        vae = DMVAE_VAE(model_size='large', z_channels=32).to(device).eval()
        if args.vae_ckpt_path:
            vae.load_pretrained(args.vae_ckpt_path)

        latent_size = args.image_size // 16
        model = DMVAE_DiT(input_size=latent_size, num_classes=args.num_classes, in_channels=32, hidden_size=1152,
                          depth=28, num_heads=16).to(device).eval()

        if args.dit_ckpt:
            ckpt = torch.load(args.dit_ckpt, map_location='cpu')
            if 'ema' in ckpt:
                model.load_state_dict(ckpt['ema'])
            elif 'model' in ckpt:
                model.load_state_dict(ckpt['model'])
            else:
                model.load_state_dict(ckpt)

        def decode_fn(z):
            z_unorm = z / args.latent_scale + args.latent_mean
            return vae.decode(z_unorm)

    else:
        if rank == 0: print("Initializing standard DiT (IGD mode)...")
        latent_size = args.image_size // 8
        model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
        state_dict = find_model(args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        diffusion = create_diffusion(str(args.num_sampling_steps))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()

        def decode_fn(z):
            return vae.decode(z / 0.18215).sample

    # --- Resources ---
    surrogate = define_model(args, args.target_nclass).to(device)
    surrogate = ReparamModule(surrogate)
    surrogate.eval()

    if rank == 0:
        print(f"Loading Surrogate Checkpoints from {args.ckpt_path}...")
    ckpts = rand_ckpts(args)
    criterion_ce = nn.CrossEntropyLoss().to(device)
    decoder_obj = DecoderWrapper(decode_fn)

    # --- Distributed Sampling Loop ---
    # Each rank generates a subset of total images
    # Assuming args.num_samples is TOTAL samples desired.
    # samples_per_rank = args.num_samples // world_size (Simplified)
    samples_per_rank = (args.num_samples + world_size - 1) // world_size

    batch_size = 1  # Keep 1 for OOM safety

    for class_label, sel_class in zip(class_labels, sel_classes):
        if rank == 0:
            os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
            print(f"Generating class: {sel_class} (idx {class_label})")

        # Barrier to ensure directory exists
        if world_size > 1: dist.barrier()

        # Calculate gradients (per rank to avoid broadcast complexity, overhead is low)
        real_gradients, cur_cls, correspond_labels = get_grads(sel_classes, class_labels, sel_class, ckpts, surrogate,
                                                               device)

        # Loop for local samples
        for i in range(samples_per_rank):
            global_idx = i * world_size + rank
            if global_idx >= args.num_samples:
                break

            # Model specific configs
            if model_type == 'igd':
                C, H, W = 4, latent_size, latent_size
            elif model_type == 'rae' and config and 'misc' in config:
                C, H, W = config['misc']['latent_size']
            elif hasattr(model, 'in_channels'):
                C = model.in_channels; H = W = latent_size
            else:
                C, H, W = 32, latent_size, latent_size

            z = torch.randn(batch_size, C, H, W, device=device)
            y = torch.tensor([class_label], device=device)

            if args.cfg_scale > 1.0:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y = torch.cat([y, y_null], 0)

            gm_resource = [decoder_obj, surrogate, ckpts, real_gradients[class_label], correspond_labels[class_label],
                           criterion_ce, args.repeat, args.repeat, args.gm_scale]
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, gm_resource=gm_resource, gen_type='igd', low=args.low,
                                high=args.high, neg_e=args.dev_scale)

            # Sample with AMP
            if model_type == 'igd':
                with torch.cuda.amp.autocast():
                    samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, clip_denoised=False,
                                                      model_kwargs=model_kwargs, progress=False, device=device)
            else:
                samples = igd_ode_sample(model, z, args.num_sampling_steps, model_kwargs, device)

            if args.cfg_scale > 1.0:
                samples, _ = samples.chunk(2, dim=0)

            samples = decode_fn(samples)

            # Save: Unique name per rank/index
            save_name = f"{global_idx + args.total_shift}.png"
            save_path = os.path.join(args.save_dir, sel_class, save_name)
            save_image(samples[0], save_path, normalize=True, value_range=(-1, 1))

            if rank == 0:
                print(f"Rank {rank}: Saved {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml')
    parser.add_argument("--model_type", type=str, choices=['igd', 'rae', 'dmvae'], default='igd')

    # Checkpoints
    parser.add_argument("--rae_ckpt", type=str, default=None)
    parser.add_argument("--dit_ckpt", type=str, default=None)
    parser.add_argument("--vae_ckpt_path", type=str, default=None)

    # Params
    parser.add_argument("--dinov2_path", type=str, default=None)
    parser.add_argument("--latent_mean", type=float, default=0.0)
    parser.add_argument("--latent_scale", type=float, default=1.0)

    # Standard IGD
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, default="mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--spec", type=str, default='none')
    parser.add_argument("--save-dir", type=str, default='../logs/test')
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--total-shift", type=int, default=0)
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--target_nclass", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--memory-size", type=int, default=64)
    parser.add_argument("--real_ipc", type=int, default=1000)
    parser.add_argument("--grad-ipc", type=int, default=80)
    parser.add_argument("--gm-scale", type=float, default=0.02)
    parser.add_argument('--dev-scale', type=float, default=0.01)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--net-type", type=str, default='convnet6')
    parser.add_argument("--low", type=int, default=500)
    parser.add_argument("--high", type=int, default=800)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--repeat", type=int, required=True)

    # device argument is no longer needed for torchrun, handled automatically

    args = parser.parse_args()
    main(args)
