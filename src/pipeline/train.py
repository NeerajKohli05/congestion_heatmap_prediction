"""
End-to-End Training Pipeline for Circuit Congestion Heatmap Prediction.
Loads data, trains Residual Attention U-Net with Hotspot-Weighted Loss, and saves checkpoints.
"""

import os
import sys
import yaml
import glob
import time
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add root directory to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, root_dir)

from src.models.unet import CongestionUNet
from src.models.losses import CompositeCongestionLoss
from src.metrics.eval_metrics import evaluate_all_metrics
from src.features.dataset import CircuitCongestionDataset, create_train_val_split
from src.data.synthetic_circuits import SyntheticCircuitGenerator




def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(pref)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()

        # Gradient clipping to prevent exploding gradients on high density spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(loader.dataset)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * features.size(0)

            # Convert to numpy for metric computation
            pred_np = predictions.cpu().numpy()
            targ_np = targets.cpu().numpy()

            for i in range(pred_np.shape[0]):
                m = evaluate_all_metrics(pred_np[i], targ_np[i])
                all_metrics.append(m)

    avg_loss = total_loss / len(loader.dataset)

    # Aggregate metric dict
    summary = {"val_loss": avg_loss}
    if all_metrics:
        for k in all_metrics[0].keys():
            summary[k] = float(np.mean([m[k] for m in all_metrics]))

    return summary


def main():
    # Load configuration
    config_path = os.path.join(root_dir, "configs/default_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set random seeds
    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device(cfg["experiment"]["device"])
    print(f"[*] Training on device: {device}")

    # Ensure processed data exists; if not, generate synthetic designs
    data_dir = os.path.join(root_dir, cfg["data"]["data_dir"])
    os.makedirs(data_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(data_dir, "*.npz"))

    if len(existing_files) < 10:
        num_to_gen = cfg["data"]["num_synthetic_samples"]
        print(f"[*] Generating {num_to_gen} synthetic circuit designs for training...")
        generator = SyntheticCircuitGenerator(
            grid_h=cfg["data"]["grid_h"],
            grid_w=cfg["data"]["grid_w"],
            random_seed=seed
        )
        generator.generate_dataset_split(num_samples=num_to_gen, output_dir=data_dir)
        print(f"[✓] Generated and saved dataset to {data_dir}")

    # Train / Validation Split
    train_files, val_files = create_train_val_split(data_dir, val_ratio=cfg["data"]["val_ratio"], seed=seed)
    print(f"[*] Dataset split: {len(train_files)} training samples, {len(val_files)} validation samples")

    train_ds = CircuitCongestionDataset(train_files, augment=True)
    val_ds = CircuitCongestionDataset(val_files, augment=False)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model & Optimization
    model = CongestionUNet(
        in_channels=cfg["data"]["in_channels"],
        out_channels=cfg["data"]["out_channels"],
        base_channels=cfg["model"]["base_channels"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] CongestionUNet initialized with {total_params:,} trainable parameters")

    criterion = CompositeCongestionLoss(
        hotspot_threshold=cfg["training"]["hotspot_threshold"],
        hotspot_weight=cfg["training"]["hotspot_weight"],
        lambda_ssim=cfg["training"]["lambda_ssim"],
        lambda_pcc=cfg["training"].get("lambda_pcc", 0.20),
        in_channels=cfg["data"]["out_channels"]
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-5)

    # Checkpoint directory
    save_dir = os.path.join(root_dir, cfg["training"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)
    best_score = -1.0
    best_ckpt_path = os.path.join(save_dir, "best_congestion_unet.pt")

    print("\n" + "=" * 75)
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val PCC':^9} | {'Hotspot F1':^10} | {'NMSE':^8}")
    print("=" * 75)

    # Training loop
    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_metrics["val_loss"]
        val_pcc = val_metrics.get("mean_pcc", 0.0)
        hotspot_f1 = val_metrics.get("mean_hotspot_f1_top10pct", 0.0)
        val_nmse = val_metrics.get("mean_nmse", 1.0)
        elapsed = time.time() - t0

        print(f"{epoch:^6d} | {train_loss:^12.4f} | {val_loss:^10.4f} | {val_pcc:^9.4f} | {hotspot_f1:^10.4f} | {val_nmse:^8.4f} ({elapsed:.1f}s)")

        # Save best model based on balanced score: val_pcc + hotspot_f1
        score = val_pcc + hotspot_f1
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pcc": val_pcc,
                "hotspot_f1": hotspot_f1,
                "val_nmse": val_nmse,
                "config": cfg
            }, best_ckpt_path)

    print("=" * 75)
    print(f"[✓] Training complete! Best validation composite score: {best_score:.4f}")
    print(f"[✓] Best model checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
