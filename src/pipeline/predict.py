"""
Inference and Visualization Script for Congestion Heatmap Prediction.
Loads a trained checkpoint, generates predictions, and saves comparative heatmap figures.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

# Add root directory to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, root_dir)

import torch
from src.models.unet import CongestionUNet
from src.metrics.eval_metrics import evaluate_all_metrics
from src.data.synthetic_circuits import SyntheticCircuitGenerator


def visualize_prediction(
    features: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    title: str = "Congestion Heatmap Prediction"
):
    """
    Renders comparative 2x3 panel:
    Top Row: Pin Density | Total RUDY Wire Demand | Macro Blockage
    Bottom Row: Ground Truth H-Congestion | Predicted H-Congestion | Absolute Error Map
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Row 0: Features
    # Pin Density (channel 0)
    im0 = axes[0, 0].imshow(features[0], cmap="plasma", origin="lower")
    axes[0, 0].set_title("Input: Pin Density")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # RUDY Total (channel 4)
    im1 = axes[0, 1].imshow(features[4], cmap="viridis", origin="lower")
    axes[0, 1].set_title("Input: Total RUDY Demand")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Macro Mask (channel 5)
    im2 = axes[0, 2].imshow(features[5], cmap="gray", origin="lower")
    axes[0, 2].set_title("Input: Macro Blockage Mask")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 1: Target vs Prediction (H-Overflow channel 0)
    vmax = max(float(np.max(target[0])), float(np.max(prediction[0])), 1.0)

    # Ground Truth
    im3 = axes[1, 0].imshow(target[0], cmap="hot", origin="lower", vmin=0, vmax=vmax)
    axes[1, 0].set_title("Ground Truth (H-Congestion)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Prediction
    im4 = axes[1, 1].imshow(prediction[0], cmap="hot", origin="lower", vmin=0, vmax=vmax)
    axes[1, 1].set_title("Predicted (CongestionUNet)")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Error Map
    error_map = np.abs(prediction[0] - target[0])
    im5 = axes[1, 2].imshow(error_map, cmap="inferno", origin="lower")
    axes[1, 2].set_title("Absolute Error Map")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Heatmap comparison saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run congestion heatmap inference")
    parser.add_argument("--checkpoint", type=str, default="./outputs/checkpoints/best_congestion_unet.pt")
    parser.add_argument("--sample_path", type=str, default=None, help="Path to .npz design file")
    parser.add_argument("--output_img", type=str, default="./outputs/visualizations/prediction_sample.png")
    args = parser.parse_args()

    ckpt_path = os.path.join(root_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found at {ckpt_path}. Please run train.py first.")
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint.get("config", {})

    # Load model
    model = CongestionUNet(
        in_channels=cfg.get("data", {}).get("in_channels", 6),
        out_channels=cfg.get("data", {}).get("out_channels", 2),
        base_channels=cfg.get("model", {}).get("base_channels", 32)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load or generate a sample
    if args.sample_path and os.path.exists(args.sample_path):
        data = np.load(args.sample_path)
        features = data["features"]
        target = data["target"]
    else:
        print("[*] Generating a fresh unseen test circuit design...")
        generator = SyntheticCircuitGenerator(
            grid_h=cfg.get("data", {}).get("grid_h", 128),
            grid_w=cfg.get("data", {}).get("grid_w", 128),
            random_seed=999
        )
        sample = generator.generate_single_design(num_cells=1800, num_nets=2400)
        features = sample["features"]
        target = sample["target"]

    # Run inference
    with torch.no_grad():
        x_tensor = torch.from_numpy(features).unsqueeze(0)  # (1, 6, H, W)
        pred_tensor = model(x_tensor)
        prediction = pred_tensor.squeeze(0).numpy()

    # Evaluate metrics
    metrics = evaluate_all_metrics(prediction, target)
    print("\n--- Test Sample Performance ---")
    print(f"Mean Pearson Correlation (PCC): {metrics['mean_pcc']:.4f}")
    print(f"Mean NMSE:                     {metrics['mean_nmse']:.4f}")
    print(f"Top 5% Hotspot F1 Score:       {metrics['mean_hotspot_f1_top5pct']:.4f}")

    # Visualize and save
    out_path = os.path.join(root_dir, args.output_img) if not os.path.isabs(args.output_img) else args.output_img
    visualize_prediction(features, target, prediction, out_path)


if __name__ == "__main__":
    main()
