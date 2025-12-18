# RAE+IGD
使用IGD改进RAE

# 环境
下包
``` shell
conda create -n raeigd python=3.12 -y
conda activate raeigd
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb
uv pip install "numpy<2" transformers einops omegaconf efficientnet_pytorch matplotlib diffusers
pip install -r requirements.txt
```
下载模型（21G）, 需要设置代理, 有一些代码运行时需要在huggingface上下载模型，需要设置代理
``` shell
uv pip install huggingface_hub
python scripts/download.py
```
# 实验结果
## ImageNet
## ResNet-10
| method  | ipc | train accuracy | validate accuracy |
|---------|-----|----------------|-------------------|
| RAE     | 10  | 78%            | 0.5428%           |
|         | 50  | 69.4           | 31%               |
|         | 100 |                |                   |
| IGD     | 10  |                |                   |
|         | 50  |                |                   |
|         | 100 |                |                   |
| DM-VAE  | 10  |                |                   |
|         | 50  |                |                   |
|         | 100 |                |                   |
| RAE+IGD | 10  |                |                   |
|         | 50  |                |                   |
|         | 100 |                |                   |

**RAE-IPC50**
```shell
python IGD/train.py -d imagenet --imagenet_dir imagenet/ipc_50/ /data2/wlf/datasets/imagenet/ -n resnet --depth 10 --nclass 1000 --norm_type instance --ipc 50 --tag test --slct_type random --spec 1k --batch_size 128 --verbose
```
![curve_0.png](https://youke2.picui.cn/s1/2025/12/16/6940c018cb9aa.png)
## DMVAE
### tokenizer
需要重新组织一些imagenet的验证集，变成ImageFolder的结构
cd到imagenet的路径中

```shell
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash```
```

```shell
bash DMVAE/scripts/train_tokenizer.sh
```
可能遇到一些错误：
gpu版本太新，.so文件找不到，那就先find，再把路径加到PATH里
```shell
find /root/miniconda3/envs/raeigd -name "libnvrtc-builtins.so*"
export LD_LIBRARY_PATH=/root/miniconda3/envs/raeigd/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```


# 改进位置