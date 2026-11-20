"""
EDA Baseline Comparison: Analytical RUDY vs. Deep Learning Model.
Evaluates how much the Residual Attention U-Net improves over traditional
analytical wire density estimators (RUDY) in predicting routing congestion.
"""

import os
import sys
import glob
import yaml
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, root_dir)

from src.models.unet import CongestionUNet
from src.metrics.eval_metrics import evaluate_all_metrics
from src.features.dataset import create_train_val_split


def run_benchmark():
    config_path = os.path.join(root_dir, "configs/default_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = os.path.join(root_dir, cfg["data"]["data_dir"])
    ckpt_path = os.path.join(root_dir, cfg["training"]["save_dir"], "best_congestion_unet.pt")

    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found at: {ckpt_path}")
        return

    # Load validation split
    _, val_files = create_train_val_split(data_dir, val_ratio=cfg["data"]["val_ratio"], seed=cfg["experiment"]["seed"])
    print(f"[*] Benchmarking on {len(val_files)} test/validation circuit samples...")

    # Load Model
    device = torch.device("cpu")
    model = CongestionUNet(
        in_channels=cfg["data"]["in_channels"],
        out_channels=cfg["data"]["out_channels"],
        base_channels=cfg["model"]["base_channels"]
    ).to(device)

    chk = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(chk["model_state_dict"] if "model_state_dict" in chk else chk)
    model.eval()

    rudy_metrics_list = []
    ml_metrics_list = []

    for fpath in val_files:
        d = np.load(fpath)
        feat = d["features"]  # (6, 128, 128)
        target = d["target"]  # (2, 128, 128)

        # 1. Classical Analytical Baseline: RUDY wire demand
        # Channel 2: Short RUDY (Horizontal demand proxy)
        # Channel 3: Long RUDY (Vertical demand proxy)
        # Channel 4: Total RUDY
        rudy_h = feat[4] # Total RUDY normalized
        rudy_v = feat[4]
        rudy_pred = np.stack([rudy_h, rudy_v], axis=0)
        m_rudy = evaluate_all_metrics(rudy_pred, target)
        rudy_metrics_list.append(m_rudy)

        # 2. Deep Learning Model Inference
        feat_t = torch.from_numpy(feat).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_t = model(feat_t).squeeze(0).cpu().numpy()
        pred_t = np.clip(pred_t, 0.0, 1.0)
        m_ml = evaluate_all_metrics(pred_t, target)
        ml_metrics_list.append(m_ml)

    # Average metrics
    rudy_avg = {k: float(np.mean([m[k] for m in rudy_metrics_list])) for k in rudy_metrics_list[0].keys()}
    ml_avg = {k: float(np.mean([m[k] for m in ml_metrics_list])) for k in ml_metrics_list[0].keys()}

    report = []
    report.append("=" * 80)
    report.append("   EDA CONGESTION BENCHMARK: ANALYTICAL RUDY vs. RESIDUAL ATTENTION U-NET")
    report.append("=" * 80)
    report.append(f"{'Metric':<32} | {'Analytical RUDY':<18} | {'AI/ML U-Net':<18} | {'Improvement':<12}")
    report.append("-" * 80)

    key_metrics = [
        ("mean_pcc", "Mean PCC (↑ correlation)", True),
        ("mean_nmse", "Mean NMSE (↓ norm error)", False),
        ("mean_mae", "Mean MAE (↓ error)", False),
        ("mean_rmse", "Mean RMSE (↓ error)", False),
        ("mean_ssim", "Mean SSIM (↑ structure)", True),
        ("mean_hotspot_f1_top10pct", "Hotspot F1 @ Top-10% (↑)", True),
        ("mean_hotspot_iou_top10pct", "Hotspot IoU @ Top-10% (↑)", True),
        ("H_Congestion_pcc", "Horizontal Congestion PCC", True),
        ("V_Congestion_pcc", "Vertical Congestion PCC", True),
    ]

    for k, label, higher_is_better in key_metrics:
        r_val = rudy_avg.get(k, 0.0)
        m_val = ml_avg.get(k, 0.0)

        if higher_is_better:
            delta = m_val - r_val
            pct = ((m_val - r_val) / (abs(r_val) + 1e-6)) * 100
            diff_str = f"+{delta:.4f} ({pct:+.1f}%)" if delta >= 0 else f"{delta:.4f} ({pct:+.1f}%)"
        else:
            delta = r_val - m_val
            pct = ((r_val - m_val) / (abs(r_val) + 1e-6)) * 100
            diff_str = f"-{m_val - r_val:.4f} ({pct:+.1f}%)"

        report.append(f"{label:<32} | {r_val:^18.4f} | {m_val:^18.4f} | {diff_str:<12}")

    report.append("=" * 80)
    report_text = "\n".join(report)
    print("\n" + report_text)

    out_file = os.path.join(root_dir, "outputs", "test_results", "eda_benchmark_comparison.txt")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write(report_text)
    print(f"\n[✓] Benchmark comparison saved to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
