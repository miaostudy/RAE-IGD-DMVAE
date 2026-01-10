import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder  # 引入标准ImageFolder用于加载验证集
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# 导入项目模块 (假设在项目根目录运行或已设置PYTHONPATH)
from data import ImageFolder_mp, CIFAR10_mp
from train_models import resnet as RN  # 使用IGD定义的ResNet
from diffusers.models import AutoencoderKL  # 严格参考IGD导入

# 设置代理 (保留您的设置)
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'


# --------------------------------------------------------
# 0. 辅助函数: 验证模型准确率
# --------------------------------------------------------
def validate(model, dataloader, device, desc="Validation"):
    """
    计算模型在给定数据加载器上的准确率
    """
    model.eval()
    correct = 0
    total = 0

    # 如果没有验证集，直接返回
    if dataloader is None:
        return 0.0

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    acc = 100. * correct / total
    model.train()  # 恢复训练模式 (注意：如果是冻结模型外部需再次eval)
    return acc


# --------------------------------------------------------
# 1. 定义 Feature Predictor M(z)
# --------------------------------------------------------
class FeaturePredictor(nn.Module):
    def __init__(self, in_channels=4, feature_dim=512):
        super().__init__()
        # 将 (B, 4, 32, 32) 映射到 (B, feature_dim)
        # 结构保持轻量，使用卷积提取空间信息后映射
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 16x16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # 8x8
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),  # 4x4
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        return self.net(x)


def get_resnet_feature_dim(depth):
    # 根据 ResNet 深度返回特征维度，参考 IGD/train_models/resnet.py
    if depth in [10, 18, 34]: return 512  # BasicBlock usually 512
    if depth in [50, 101, 152]: return 2048  # Bottleneck usually 2048
    return 512


# --------------------------------------------------------
# 2. 主函数
# --------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- A. 准备数据 (训练集 & 验证集) ---
    print(f"=> Preparing dataset: {args.dataset}")
    # 数据增强：训练M(z)时可以使用简单的变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. 训练集加载器
    if args.dataset == 'cifar10':
        dataset = CIFAR10_mp(args.data_path, train=True, transform=transform, download=True)
        n_classes = 10
    else:
        dataset = ImageFolder_mp(args.data_path, transform=transform, nclass=args.nclass,
                                 ipc=args.real_ipc, spec=args.spec, phase=args.phase, seed=0)
        n_classes = args.nclass

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2. 验证集加载器 (新增)
    val_loader = None
    if args.dataset == 'cifar10':
        # CIFAR10 使用 train=False 作为测试/验证集
        val_dataset = CIFAR10_mp(args.data_path, train=False, transform=transform, download=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print("=> Validation set loaded (CIFAR10 Test set)")
    else:
        # ImageNet/ImageNette 尝试加载 'val' 文件夹
        val_dir = '/data/wlf/datasets/imagenette/val'
        if os.path.exists(val_dir):
            # 使用标准 ImageFolder 加载验证集
            # 注意：如果训练集是 ImageNet 的子集 (如 ImageNette)，请确保 val 文件夹只包含对应的类别
            val_dataset = ImageFolder(root=val_dir, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            print(f"=> Validation set loaded from {val_dir}")
        else:
            print(f"Warning: Validation directory not found at {val_dir}. Skipping validation.")

    # --- B. 加载模型 ---

    # 1. 加载 VAE (Encoder)
    print(f"=> Loading VAE: stabilityai/sd-vae-ft-{args.vae}")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    for param in vae.parameters(): param.requires_grad = False

    # 2. 加载或训练 ResNet (Downstream Classifier / Surrogate)
    print(f"=> Loading ResNet-{args.depth} (Surrogate)")
    resnet = RN.ResNet(args.dataset, args.depth, n_classes, norm_type=args.norm_type, size=256).to(device)

    if args.resnet_ckpt and os.path.exists(args.resnet_ckpt):
        print(f"Loading ResNet checkpoint from {args.resnet_ckpt}")
        state_dict = torch.load(args.resnet_ckpt, map_location=device)
        resnet.load_state_dict(state_dict, strict=False)

        # 加载后立即验证一次
        if val_loader:
            acc = validate(resnet, val_loader, device, desc="Checking Loaded Checkpoint")
            print(f"=> Loaded ResNet Accuracy: {acc:.2f}%")

    else:
        # 完整的微调逻辑
        print(f"Warning: No ResNet checkpoint provided at '{args.resnet_ckpt}'.")
        print(f"Starting fine-tuning ResNet from scratch for {args.finetune_epochs} epochs...")

        resnet.train()
        optimizer_ft = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)
        criterion_ce = nn.CrossEntropyLoss()

        best_acc = 0.0

        for ft_epoch in range(args.finetune_epochs):
            total_loss = 0
            correct = 0
            total = 0
            pbar = tqdm(dataloader, desc=f"Fine-tuning ResNet Epoch {ft_epoch + 1}/{args.finetune_epochs}")

            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                optimizer_ft.zero_grad()
                outputs = resnet(x)
                loss = criterion_ce(outputs, y)
                loss.backward()
                optimizer_ft.step()

                # 统计信息
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'train_acc': 100. * correct / total})

            # Epoch 结束，进行验证
            val_acc = 0.0
            if val_loader:
                val_acc = validate(resnet, val_loader, device, desc="Validating")
                tqdm.write(f"Epoch {ft_epoch + 1} finished. Val Acc: {val_acc:.2f}%")

            scheduler_ft.step()

            # 保存最佳模型或最新模型
            if val_acc >= best_acc:
                best_acc = val_acc
                ft_save_path = os.path.join(args.save_dir, f"resnet{args.depth}_finetuned.pth")
                torch.save(resnet.state_dict(), ft_save_path)

        print(f"Fine-tuned ResNet saved to {ft_save_path} (Best Acc: {best_acc:.2f}%)")
        # 重新加载最佳权重
        resnet.load_state_dict(torch.load(ft_save_path))

    # 冻结 ResNet
    resnet.eval()
    for param in resnet.parameters(): param.requires_grad = False

    # --- C. 训练 Feature Predictor M(z) ---
    feature_dim = get_resnet_feature_dim(args.depth)
    print(f"=> Building M(z) with output dim {feature_dim}")

    model_M = FeaturePredictor(feature_dim=feature_dim).to(device)
    optimizer_M = optim.Adam(model_M.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()

    print("=> Start Training M(z)...")
    for epoch in range(args.epochs):
        model_M.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training M(z) Epoch {epoch + 1}/{args.epochs}")

        for x, _ in pbar:
            x = x.to(device)

            with torch.no_grad():
                # 1. 获取真实特征 (Target)
                # ResNet.get_feature 返回 list，[-2] 是 avgpool 后的特征 (B, Dim)
                real_feats = resnet.get_feature(x, idx_from=0)[-2]

                # 2. 获取 Latent z
                # DiT 训练时使用的是缩放后的 latent (z * 0.18215)
                # 为了让 M(z) 能直接用于 Diffusion 过程中的 latent，我们也应该用缩放后的 z 训练
                dist = vae.encode(x).latent_dist
                z = dist.sample().mul_(0.18215)  # Scale factor from sample_mp.py logic

            # 3. 预测
            pred_feats = model_M(z)

            loss = criterion_mse(pred_feats, real_feats)

            optimizer_M.zero_grad()
            loss.backward()
            optimizer_M.step()

            total_loss += loss.item()
            pbar.set_postfix({'mse': loss.item()})

    # 保存 M(z)
    save_path = os.path.join(args.save_dir, "feature_predictor_M.pth")
    torch.save(model_M.state_dict(), save_path)
    print(f"=> Saved M(z) to {save_path}")

    # --- D. 构建特征库 (Feature Bank) ---
    print("=> Building Real Feature Bank...")
    feature_bank = defaultdict(list)

    # 使用不 shuffle 的 loader
    extract_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        for x, y in tqdm(extract_loader, desc="Extracting features"):
            x = x.to(device)
            # 提取真实特征
            feats = resnet.get_feature(x, idx_from=0)[-2].cpu()
            labels = y.cpu().numpy()

            for i, label in enumerate(labels):
                feature_bank[int(label)].append(feats[i])

    # 转为 Tensor 并保存
    final_bank = {k: torch.stack(v) for k, v in feature_bank.items()}
    bank_path = os.path.join(args.save_dir, "feature_bank.pt")
    torch.save(final_bank, bank_path)
    print(f"=> Saved Feature Bank to {bank_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--save_dir", type=str, default='./checkpoints_proxy')
    parser.add_argument("--vae", type=str, default="mse", choices=["ema", "mse"])
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--resnet_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for training M(z)")
    parser.add_argument("--finetune_epochs", type=int, default=100,
                        help="Epochs for fine-tuning ResNet if no ckpt provided")

    # Dataset args
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--real_ipc", type=int, default=1000)
    parser.add_argument("--spec", type=str, default='none')
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--norm_type", type=str, default='instance')

    args = parser.parse_args()
    main(args)