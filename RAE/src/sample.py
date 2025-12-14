# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch.nn as nn
import math
from time import time
import argparse
from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs
from stage1 import RAE
from torchvision.utils import save_image
import torch
import sys
import os
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

from stage2.models import Stage2ModelProtocol
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def sample_one_image(class_labels, save_path, dynamic_seed, misc,device,guidance_config,sample_fn,model,rae):
    # Setup PyTorch:
    torch.manual_seed(dynamic_seed)
    torch.cuda.manual_seed(dynamic_seed)
    num_classes = misc.get("num_classes", 1000)
    print(num_classes)
    latent_size = misc.get("latent_size", (768, 16, 16))
    # Labels to condition the models with (feel free to change):

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, *latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)

    # set guidance setup
    guidance_scale = guidance_config.get("scale", 1.0)
    if guidance_scale > 1.0:
        t_min, t_max = guidance_config.get("t_min", 0.0), guidance_config.get("t_max", 1.0)
        model_kwargs = dict(y=y, cfg_scale=guidance_scale,
                            cfg_interval=(t_min, t_max))
        guidance_method = guidance_config.get("method", "cfg")
        if guidance_method == "autoguidance":
            guid_model_config = guidance_config.get("guidance_model", None)
            assert guid_model_config is not None, "Please provide a guidance models config when using autoguidance."
            guid_model: Stage2ModelProtocol = instantiate_from_config(guid_model_config).to(device)
            guid_model.eval()  # important!
            guid_fwd = guid_model.forward
            model_kwargs['additional_model_forward'] = guid_fwd
            model_fwd = model.forward_with_autoguidance
        else:
            model_fwd = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_fwd = model.forward
    # Sample images:
    start_time = time()
    samples: torch.Tensor = sample_fn(z, model_fwd, **model_kwargs)[-1]
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample
    samples = rae.decode(samples)
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, save_path, nrow=4, normalize=True, value_range=(0, 1))


def main(args):
    classes_name = {}
    with open('IGD/misc/class_indices.txt', "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            classes_name[i] = line.strip()

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载参数
    rae_config, model_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(args.config)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()  # important!
    rae.eval()
    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(
        shift_dim / shift_base)
    print(
        f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")
    transport = create_transport(
        **transport_config['params'],
        time_dist_shift=time_dist_shift
    )
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config['mode'], sampler_config['params']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            **sampler_params
        )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            **sampler_params,
            # sampling_method=args.sampling_method,
            # diffusion_form=args.diffusion_form,
            # diffusion_norm=args.diffusion_norm,
            # last_step=args.last_step,
            # last_step_size=args.last_step_size,
            # num_steps=args.num_sampling_steps,
        )
    else:
        raise NotImplementedError(f"Invalid sampling mode {mode}.")

    output = os.path.join(args.output, str(args.ipc))

    for class_id in range(1000):
        save_dir = os.path.join(output, str(classes_name[class_id]))
        os.makedirs(save_dir, exist_ok=True)
        for i in range(args.ipc):
            dynamic_seed = args.seed + class_id * 1000 + i + int(time() * 1000) % 1000000
            sample_one_image([class_id], os.path.join(save_dir, f'{i}.png'),dynamic_seed,misc,device,guidance_config,sample_fn,model,rae)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='RAE/configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml',type=str,
                        help="Path to the config file.")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--output", type=str, default='imagenet')
    args = parser.parse_known_args()[0]
    main(args)
