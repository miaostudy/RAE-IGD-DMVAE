from torchvision.datasets.utils import download_url
import torch
import os
from huggingface_hub import hf_hub_download, snapshot_download
import argparse

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

pretrained_models = ['sen-ye/dmvae', 'nyu-visionx/RAE-collections', 'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt']


def find_model(model_name, args):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        if len(model_name.split('/')) == 2:
            download_model_from_hunggingface(model_name, args.output)
        else:
            download_model(model_name, args.output)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]

def download_model_from_hunggingface(model_name, path):
    snapshot_download(repo_id=model_name, local_dir=path)


def download_model(model_name, path):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = os.path.join(path, model_name)
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, 'pretrained_models')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--output', default='pretrained_models', type=str)
    args = parser.parse_args()
    # Download all DiT checkpoints
    for model in pretrained_models:
        find_model(model, args)
    print('Done.')
