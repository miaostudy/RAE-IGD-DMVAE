import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# 假设你的项目路径已经在 sys.path 中
# 引入你的 RAE 或 DMVAE 模型定义
# 根据你的文件结构，这里以 RAE 为例，你也可能用 DMVAE.dmvae_models.vae
from RAE.src.stage1.rae import RAE


# from DMVAE.dmvae_models.vae import VAE # 如果是用 DMVAE

# ==========================================
# 1. 定义 Latent Proxy (轻量级分类器)
# ==========================================
class LatentProxy(nn.Module):
    def __init__(self, input_dim, num_classes=1000):
        super().__init__()
        # DINOv2-Base 的 hidden_size 通常是 768
        # 根据你的 RAE/VAE 配置调整 input_dim
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
        # x shape: [B, C, H, W] or [B, N, C] -> 需要 flatten
        x = x.flatten(start_dim=1)
        return self.net(x)


# ==========================================
# 2. 辅助函数：提取 Latent 数据集
# ==========================================
@torch.no_grad()
def extract_latents(model, data_loader, device):
    print("Extracting latents from training data...")
    model.eval()
    latents_list = []
    labels_list = []

    for x, y in tqdm(data_loader):
        x = x.to(device)
        # RAE 的 encode 方法通常返回 latent code
        # 注意：这里需要确认 encode 是否包含了 normalize 等预处理
        # 你的 RAE.encode 看起来已经包含了处理
        z = model.encode(x)

        # 如果 z 是 [B, C, H, W]，我们在这里不需要变，交给 Proxy 去 flatten
        latents_list.append(z.cpu())
        labels_list.append(y)

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    print(f"Latents extracted. Shape: {latents.shape}")
    return latents, labels


# ==========================================
# 3. 训练 Latent Proxy
# ==========================================
def train_proxy(latents, labels, device, num_classes=1000, epochs=10):
    print("Training Latent Proxy...")
    # 计算 input_dim (flatten 后)
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

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f} | Acc: {correct / total:.4f}")

    return proxy


# ==========================================
# 4. 核心实验：Sanity Check
# ==========================================
def run_sanity_check(rae_model, latent_proxy, pixel_classifier, val_loader, device, class_names=None):
    print("\nRunning Sanity Check: Latent Guidance -> Pixel Eval")
    rae_model.eval()
    latent_proxy.eval()
    pixel_classifier.eval()

    # 获取一个 Batch 的数据
    x_real, y_real = next(iter(val_loader))
    x_real, y_real = x_real.to(device), y_real.to(device)

    # 1. 得到真实的 Latent
    with torch.no_grad():
        z_real = rae_model.encode(x_real)

    # 2. 扰动 Latent (模拟扩散过程中的中间态)
    # 加一点噪声，让它稍微偏离最优解，这样梯度才有优化的空间
    noise = torch.randn_like(z_real) * 0.5
    z_noisy = z_real + noise

    # 开启梯度以进行优化
    z_optimized = z_noisy.clone().detach().requires_grad_(True)

    # 3. Latent Space 梯度引导 (IGD Step in Latent)
    optimizer_z = optim.SGD([z_optimized], lr=0.1)  # 学习率（指导强度）可能需要调整

    print("Optimizing Latent Code...")
    for step in range(5):  # 迭代几步，模拟多次指导
        optimizer_z.zero_grad()
        # 计算 Latent Proxy 的 loss
        logits = latent_proxy(z_optimized)

        # 我们希望 z_optimized 被分类为 y_real
        # 这里使用 target class 的梯度来引导
        # 也可以使用 CrossEntropyLoss
        loss = -logits[range(len(y_real)), y_real].sum()  # Maximize logit of target class

        loss.backward()

        # 打印一下梯度的模长，确保有梯度传回来
        grad_norm = z_optimized.grad.norm().item()

        optimizer_z.step()

    print(f"Optimization done. Grad norm at last step: {grad_norm:.4f}")

    # 4. 解码 (Decode)
    with torch.no_grad():
        # 解码 优化前 的 noisy latent
        x_noisy_recon = rae_model.decode(z_noisy)
        # 解码 优化后 的 optimized latent
        x_opt_recon = rae_model.decode(z_optimized)

    # 5. 像素级评估 (ResNet Eval)
    # 注意：ResNet 需要特定的 normalize，这里假设 x 已经是 [0,1] 或 [-1,1]，需要适配 ResNet
    # 通常 ResNet 需要 normalize mean=[0.485, ...], std=[0.229, ...]
    # 这里为了演示简单，假设 x 已经适配或 ResNet 鲁棒。实际使用建议加上 transform。

    # 定义 ResNet 的 Normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def preprocess_for_resnet(img):
        # 假设 img 是 [-1, 1] -> [0, 1]
        img = (img + 1) / 2.0
        img = torch.clamp(img, 0, 1)
        # Apply normalize per image
        return torch.stack([normalize(im) for im in img])

    x_noisy_input = preprocess_for_resnet(x_noisy_recon)
    x_opt_input = preprocess_for_resnet(x_opt_recon)

    with torch.no_grad():
        pred_noisy = pixel_classifier(x_noisy_input)
        pred_opt = pixel_classifier(x_opt_input)

        prob_noisy = torch.softmax(pred_noisy, dim=1)
        prob_opt = torch.softmax(pred_opt, dim=1)

    # 6. 统计结果
    improved_count = 0
    total_count = len(y_real)

    print("\n--- Result Comparison ---")
    for i in range(min(5, total_count)):  # 打印前5个样本的详情
        target = y_real[i].item()
        conf_noisy = prob_noisy[i, target].item()
        conf_opt = prob_opt[i, target].item()

        status = "IMPROVED" if conf_opt > conf_noisy else "DEGRADED"
        print(f"Sample {i} (Class {target}): Noisy Conf={conf_noisy:.4f} -> Opt Conf={conf_opt:.4f} [{status}]")

        if conf_opt > conf_noisy:
            improved_count += 1

    print(f"\nSummary: {improved_count}/{total_count} samples improved in Pixel Space confidence.")

    if improved_count > total_count * 0.8:
        print("\n✅ 结论：假设成立！Latent Space 的梯度引导成功迁移到了 Pixel Space。")
        print("建议：可以大胆将 IGD 迁移到 Latent Space。")
    else:
        print("\n❌ 结论：假设可能不成立或存在 Gap。")
        print("可能原因：Decoder 丢失了纹理特征，或者 ResNet 和 Latent Proxy 关注点完全不同。")
        print("建议：检查 Decoder 重建质量，或在 Guidance 中加入特征匹配项。")


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. 准备数据和模型 (根据你的环境修改)
    # 假设你有一个函数 load_dataset() 返回 dataloader
    # from IGD.data import load_data
    # args = ...
    # _, train_loader, val_loader, num_classes = load_data(args)

    # 模拟数据 (For Debugging)
    print("Generating dummy data for testing code logic...")
    dummy_x = torch.randn(100, 3, 224, 224).to(device)
    dummy_y = torch.randint(0, 10, (100,)).to(device)
    train_loader = [(dummy_x, dummy_y)]
    val_loader = [(dummy_x, dummy_y)]
    num_classes = 10

    # 加载 RAE
    print("Loading RAE Model...")


    # 请替换为实际的加载代码
    # rae = RAE(encoder_cls='Dinov2withNorm', ...).to(device)
    # rae.load_state_dict(torch.load('path/to/ckpt'))

    # 这是一个 Mock 类，用于跑通上面的逻辑，实际使用请替换为真模型
    class MockRAE(nn.Module):
        def encode(self, x): return torch.randn(x.size(0), 768, 16, 16).to(x.device)  # Mock Latent

        def decode(self, z): return torch.randn(z.size(0), 3, 224, 224).to(z.device)  # Mock Recon


    rae = MockRAE().to(device)

    # 加载 ResNet (Pixel Classifier)
    print("Loading Pixel Classifier (ResNet)...")
    pixel_classifier = models.resnet18(pretrained=False)  # 实际实验请设为 True
    pixel_classifier.fc = nn.Linear(512, num_classes)
    pixel_classifier.to(device)

    # B. 提取 Latent 并训练 Proxy
    latents, labels = extract_latents(rae, train_loader, device)
    latent_proxy = train_proxy(latents, labels, device, num_classes=num_classes)

    # C. 运行验证
    run_sanity_check(rae, latent_proxy, pixel_classifier, val_loader, device)