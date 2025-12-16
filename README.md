# RAE+IGD
使用IGD改进RAE

# 环境
下包
``` shell
conda create -n raeigd python=3.12 -y
conda activate raeigd
pip install uv
uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb
uv pip install "numpy<2" transformers einops omegaconf efficientnet_pytorch matplotlib diffusers

```
下载模型（21G）, 需要设置代理, 有一些代码运行时需要在huggingface上下载模型，需要设置代理
```shell
cd RAE
pip install huggingface_hub
hf download nyu-visionx/RAE-collections --local-dir models
python IGD/download.py
```
# 实验结果
## ImageNet
## ResNet-10
| method  | ipc | train accuracy | validate accuracy |
|---------|-----|----------------|-------------------|
| RAE     | 10  | 78%            |       0.5428%     |
|         | 50  |                |                   |
|         | 100 |                |                   |
| IGD     | 10  |                |                   |
|         | 50  |                |                   |
|         | 100 |                |                   |
| RAE+IGD | 10  |                |                   |
|         | 50  |                |                   |
|         | 100 |                |                   |

**RAE-IPC10**
```shell
python IGD/train.py -d imagenet --imagenet_dir imagenet/ipc_50/ /data2/wlf/datasets/imagenet/ -n resnet --depth 10 --nclass 1000 --norm_type instance --ipc 50 --tag test --slct_type random --spec 1k --batch_size 128 --verbose
```
![curve_0.png](https://youke2.picui.cn/s1/2025/12/16/6940c018cb9aa.png)

# 改进位置