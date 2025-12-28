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
# Baseline
## IGD
### ImageNet1K
这个和ImageNete、ImageWoof的设置有很大不同，不能直接沿用设置
![](https://youke2.picui.cn/s1/2025/12/27/694fc73be2283.png)

相关设置：
1. 软标签生成模型：**resnet-18**
2. 生成模型：**MiniMax**，这是一个针对数据集蒸馏任务微调过的DIT模型
3. 采样步：50
4. 代理模型参数：
    ```shell
    model:ConvNet-6
    lr: 0.01 # 有一段代码是：当“args.dataset=imagenet”时lr=0.1, 而论文中明确指出是0.01来训练代理模型, 这里我选择相信代码
    epoch: 50
    ```
5. 选择有代表性的检查点：代理模型的检查点不是每个epcoh都保存的，只保存“与参考checkpoint”的梯度相似度不超过**0.7**的checkpoint。计算引导Loss的时候只是用这些checkpoint
6. $k=10$, $\gamma_t=100$

代码实现与论文描述不一致的地方：
1. 并没有实现 **Choosing representative checkpoints via gradient similarity**：
   
    在训练代理模型的时候，保存了完整的50个epoch的模型参数。在采样的时候直接**人为指定加载哪些epoch的参数**，比如`convnet6`就加载`idxs = [0,5,18,52]`时的代理模型参数来进行引导。
2. 论文里各个位置说的都是训练50个epoch，在加载idx的时候加载了52。


ckpt自己就只有30%左右的准确率，用它来算梯度引导采样，是不是不好

![](https://youke2.picui.cn/s1/2025/12/28/6950d98f2e617.png)
### MiniMax
IGD的imagenet1k基于与训练的Minimax模型，但是两者均未给出checkpoint，需要自己训练。


![](https://youke2.picui.cn/s1/2025/12/28/6950daa6e50ae.png)
单机双卡：
```shell
torchrun --nnode=1 --nproc_per_node=2 --master_port=25678 MinimaxDiffusion/train_dit.py --model DiT-XL/2   --data-path /data2/wlf/datasets/imagenet/train/ --ckpt pretrained_models/DiT-XL-2-256x256.pt --global-batch-size 8 --tag minimax --ckpt-every 12000 --log-every 1500 --epochs 8 --condense --finetune-ipc -1 --results-dir logs/run-0 --spec 1k
```

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

# 结果和命令
## ImageNet
## ResNet-10
| method         | ipc | train accuracy | validate accuracy |
|----------------|-----|----------------|-------------------|
| RAE            | 10  | 78%            | 0.5428%           |
|                | 50  | 69.4           | 31%               |
|                | 100 | -              | -                 |
| IGD            | 10  |                |                   |
|                | 50  |                |                   |
|                | 100 |                |                   |
| DM-VAE         | 10  |                |                   |
|                | 50  |                |                   |
|                | 100 |                |                   |
| RAE+IGD(fixed) | 10  |                |                   |
|                | 50  |                |                   |
|                | 100 |                |                   |


![curve_0.png](https://youke2.picui.cn/s1/2025/12/16/6940c018cb9aa.png)

**RAE-IPC50**
```shell
python IGD/train.py -d imagenet --imagenet_dir imagenet/ipc_50/ /data2/wlf/datasets/imagenet/ -n resnet --depth 10 --nclass 1000 --norm_type instance --ipc 50 --tag test --slct_type random --spec 1k --batch_size 128 --verbose
```

# 改进位置