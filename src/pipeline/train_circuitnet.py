"""
Enhanced Training Pipeline for CircuitNet Real Data.
Supports both synthetic data (auto-generated) and real CircuitNet .npz data.
Includes improved logging, mixed-precision, and LR warm-up.
"""

import os
import sys
import yaml
import glob
import time
import argparse
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
        return torch.device("cpu")
    return torch.device(pref)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None) -> float:
    model.train()
    total_loss = 0.0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:  # AMP mixed precision
            with torch.cuda.amp.autocast():
                predictions = model(features)
                loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / max(len(loader.dataset), 1)


def validate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * features.size(0)

            pred_np = predictions.cpu().numpy()
            targ_np = targets.cpu().numpy()
            for i in range(pred_np.shape[0]):
                all_metrics.append(evaluate_all_metrics(pred_np[i], targ_np[i]))

    summary = {"val_loss": total_loss / max(len(loader.dataset), 1)}
    if all_metrics:
        for k in all_metrics[0]:
            summary[k] = float(np.mean([m[k] for m in all_metrics]))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--data_source", type=str, default="auto",
                        choices=["auto", "synthetic", "circuitnet"],
                        help="'auto' uses CircuitNet if available, else synthetic")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = parser.parse_args()

    config_path = os.path.join(root_dir, args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    seed = cfg["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device(cfg["experiment"]["device"])
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"[*] Device: {device}  |  AMP: {use_amp}")

    # ── Data source selection ─────────────────────────────────────────────────
    data_dir = os.path.join(root_dir, cfg["data"]["data_dir"])
    os.makedirs(data_dir, exist_ok=True)
    existing = glob.glob(os.path.join(data_dir, "*.npz"))

    if args.data_source == "synthetic" or (args.data_source == "auto" and len(existing) < 10):
        n = cfg["data"]["num_synthetic_samples"]
        print(f"[*] Generating {n} synthetic circuits...")
        gen = SyntheticCircuitGenerator(
            grid_h=cfg["data"]["grid_h"],
            grid_w=cfg["data"]["grid_w"],
            random_seed=seed
        )
        gen.generate_dataset_split(n, data_dir)
        existing = glob.glob(os.path.join(data_dir, "*.npz"))

    source_tag = "CircuitNet" if any("nvdla" in f or "RISCY" in f or "zero" in f for f in existing) else "Synthetic"
    print(f"[*] Using {source_tag} data: {len(existing)} samples in {data_dir}")

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    train_files, val_files = create_train_val_split(data_dir, cfg["data"]["val_ratio"], seed)
    print(f"[*] Split: {len(train_files)} train / {len(val_files)} val")

    train_ds = CircuitCongestionDataset(train_files, augment=True)
    val_ds   = CircuitCongestionDataset(val_files,   augment=False)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CongestionUNet(
        in_channels=cfg["data"]["in_channels"],
        out_channels=cfg["data"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] CongestionUNet: {n_params:,} parameters")

    criterion = CompositeCongestionLoss(
        hotspot_threshold=cfg["training"]["hotspot_threshold"],
        hotspot_weight=cfg["training"]["hotspot_weight"],
        lambda_ssim=cfg["training"]["lambda_ssim"],
        in_channels=cfg["data"]["out_channels"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["training"]["epochs"], eta_min=1e-5)

    save_dir = os.path.join(root_dir, cfg["training"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)
    best_f1 = -1.0
    ckpt_path = os.path.join(save_dir, "best_congestion_unet.pt")

    # ── Training Loop ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^10} | {'PCC':^8} | {'F1@5%':^8} | {'Time':^6}")
    print("=" * 70)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        pcc = val_metrics.get("mean_pcc", 0.0)
        f1  = val_metrics.get("mean_hotspot_f1_top5pct", 0.0)
        elapsed = time.time() - t0

        print(f"{epoch:^6d} | {train_loss:^12.4f} | {val_metrics['val_loss']:^10.4f} | {pcc:^8.4f} | {f1:^8.4f} | {elapsed:^5.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pcc": pcc,
                "hotspot_f1": f1,
                "data_source": source_tag,
                "config": cfg,
            }, ckpt_path)

    print("=" * 70)
    print(f"[✓] Best Hotspot F1@5%: {best_f1:.4f}  |  checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
