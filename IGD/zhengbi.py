import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import os
import sys
import yaml
import importlib
import matplotlib.pyplot as plt

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "RAE", "src"))
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"


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


# ==========================================
# 1. Latent Proxy
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
# 2. Extract Latents
# ==========================================
@torch.no_grad()
def extract_latents(model, data_loader, device):
    print("Extracting latents from training data...")
    model.eval()
    latents_list = []
    labels_list = []

    max_batches = 200
    for i, (x, y) in enumerate(tqdm(data_loader, desc="Extracting")):
        x = x.to(device)
        z = model.encode(x)
        if isinstance(z, tuple): z = z[0]
        if hasattr(z, 'sample'): z = z.mode()

        latents_list.append(z.cpu())
        labels_list.append(y)
        if i >= max_batches: break

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return latents, labels


# ==========================================
# 3. Training Helpers
# ==========================================
def train_proxy(latents, labels, device, num_classes=1000, epochs=10):
    print(f"\n[Step 1] Training Latent Proxy (on {len(labels)} samples)...")
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
        print(f"  Proxy Epoch {epoch + 1}/{epochs} | Acc: {correct / total:.4f}")
    return proxy


def train_pixel_classifier_head(model, data_loader, device, epochs=5):
    print(f"\n[Step 2] Training Pixel Classifier Head (Linear Probe)...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    max_batches = 100

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(data_loader, desc=f"  Pixel Epoch {epoch + 1}", total=min(len(data_loader), max_batches))
        for i, (x, y) in enumerate(pbar):
            if i >= max_batches: break
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix({'Acc': f"{correct / total:.4f}"})

    return model


# ==========================================
# 4. Correlation Experiment (Noise Sweep)
# ==========================================
def run_correlation_experiment(rae_model, latent_proxy, pixel_classifier, val_loader, device,
                               save_dir="./results/correlation"):
    print("\n[Step 3] Running Correlation Experiment: Latent Acc vs Pixel Acc")
    os.makedirs(save_dir, exist_ok=True)

    rae_model.eval()
    latent_proxy.eval()
    pixel_classifier.eval()

    # 1. Collect all validation latents first (to speed up loop)
    #    We extract z_real for the whole validation set (or a large subset)
    print("  Encoding Validation Set...")
    all_z = []
    all_y = []
    max_val_batches = 50  # Adjust based on memory

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(val_loader, desc="Encoding")):
            if i >= max_val_batches: break
            x = x.to(device)
            z = rae_model.encode(x)
            if isinstance(z, tuple): z = z[0]
            if hasattr(z, 'sample'): z = z.mode()
            all_z.append(z.cpu())
            all_y.append(y)

    all_z = torch.cat(all_z, dim=0).to(device)  # Keep on GPU for speed if fits
    all_y = torch.cat(all_y, dim=0).to(device)
    print(f"  Validation Subsample Size: {len(all_y)}")

    # 2. Define Noise Scales (Sweep)
    # Range: 0 (clean) to large noise (random)
    noise_scales = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

    results = {
        "noise": [],
        "latent_acc": [],
        "pixel_acc": []
    }

    # Preprocessing for ResNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def prep(x):
        x = torch.clamp(x, 0, 1)  # Ensure valid image range
        return torch.stack([normalize(img) for img in x])

    print("\n  Starting Noise Sweep...")
    print(f"  {'Noise':<6} | {'Latent Acc':<10} | {'Pixel Acc':<10} | {'Ratio (P/L)':<10}")
    print("-" * 50)

    for scale in noise_scales:
        # Add noise
        noise = torch.randn_like(all_z) * scale
        z_noisy = all_z + noise

        # --- Measure Latent Accuracy ---
        with torch.no_grad():
            # Batch processing to avoid OOM
            l_correct = 0
            p_correct = 0
            total = len(all_y)
            batch_size = 64

            for i in range(0, total, batch_size):
                z_batch = z_noisy[i:i + batch_size]
                y_batch = all_y[i:i + batch_size]

                # Latent Pred
                l_logits = latent_proxy(z_batch)
                l_pred = l_logits.argmax(dim=1)
                l_correct += (l_pred == y_batch).sum().item()

                # --- Measure Pixel Accuracy ---
                # Decode
                x_recon = rae_model.decode(z_batch)

                # Handle RAE output range (assuming ~[0,1] or [-1,1])
                # Heuristic: if mean < 0, shift it.
                # (Ideally use same logic as training, here we assume [-1,1] -> [0,1])
                # x_recon = (x_recon + 1) / 2.0
                # OR if your model outputs [0,1] directly (like sigmoid), comment above line.
                # Based on previous output, check if clamping is enough.
                # Let's try standard un-normalization if we knew it.
                # For safety, let's just clamp if it looks like [0,1], or shift if [-1,1]
                if x_recon.min() < -0.2:
                    x_recon = (x_recon + 1) / 2.0

                # Pixel Pred
                p_logits = pixel_classifier(prep(x_recon))
                p_pred = p_logits.argmax(dim=1)
                p_correct += (p_pred == y_batch).sum().item()

            l_acc = l_correct / total
            p_acc = p_correct / total

            results["noise"].append(scale)
            results["latent_acc"].append(l_acc)
            results["pixel_acc"].append(p_acc)

            ratio = p_acc / (l_acc + 1e-6)
            print(f"  {scale:<6.1f} | {l_acc:<10.4f} | {p_acc:<10.4f} | {ratio:<10.4f}")

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(results["latent_acc"], results["pixel_acc"], 'o-', linewidth=2, color='blue',
             label='Correlation Trajectory')

    # Add annotations for noise levels
    for i, txt in enumerate(results["noise"]):
        plt.annotate(f"n={txt}", (results["latent_acc"][i], results["pixel_acc"][i]), xytext=(5, 5),
                     textcoords='offset points')

    plt.title(f"Latent Accuracy vs. Pixel Accuracy Correlation\n(Noise Sweep 0.0 -> 10.0)", fontsize=14)
    plt.xlabel("Latent Proxy Accuracy", fontsize=12)
    plt.ylabel("Pixel Classifier (ResNet) Accuracy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label="Perfect 1:1")  # Reference line
    plt.legend()

    plot_path = os.path.join(save_dir, "correlation_plot.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

    # 4. Conclusion
    correlation = np.corrcoef(results["latent_acc"], results["pixel_acc"])[0, 1]
    print(f"Pearson Correlation Coefficient: {correlation:.4f}")

    if correlation > 0.9:
        print("\n✅ 结论：极强正相关！(Pearson > 0.9)")
        print("证明：Latent空间的分类能力可以直接映射到Pixel空间的分类能力。")
    elif correlation > 0.7:
        print("\n⚠️ 结论：强相关，但存在非线性衰减。")
    else:
        print("\n❌ 结论：相关性较弱。")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    from IGD.argument import args
    from IGD.data import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    _, train_loader, val_loader, num_classes = load_data(args)

    # 2. Load RAE
    print("Loading RAE...")
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if 'stage_2' in config and 'stage2.dmvae_models' in config['stage_2'].get('target', ''):
            config['stage_2']['target'] = config['stage_2']['target'].replace('stage2.dmvae_models', 'stage2.models')

    rae = instantiate_from_config(config['stage_1']).to(device).eval()
    if args.rae_ckpt:
        print(f"Loading RAE: {args.rae_ckpt}")
        st = torch.load(args.rae_ckpt, map_location='cpu')
        if 'model' in st: st = st['model']
        rae.load_state_dict(st, strict=False)

    # 3. Setup Models
    # A. Latent Proxy (Train from scratch)
    latents, labels = extract_latents(rae, train_loader, device)
    latent_proxy = train_proxy(latents, labels, device, num_classes=num_classes, epochs=10)

    # B. Pixel Classifier (Train head from scratch)
    print("Initializing Pixel Classifier (ResNet18)...")
    pixel_classifier = models.resnet18(pretrained=True)
    pixel_classifier.fc = nn.Linear(512, num_classes)
    pixel_classifier.to(device)
    pixel_classifier = train_pixel_classifier_head(pixel_classifier, train_loader, device, epochs=3)

    # 4. Run Correlation Experiment (Replace Sanity Check)
    run_correlation_experiment(rae, latent_proxy, pixel_classifier, val_loader, device)