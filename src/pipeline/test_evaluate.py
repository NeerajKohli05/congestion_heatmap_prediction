"""
Comprehensive Testing, Accuracy Evaluation & Results Report Generator.
Runs full test suite on the trained model and produces:
  1. Per-sample metric table
  2. Per-channel breakdown (H-overflow, V-overflow)
  3. Confusion-matrix-style hotspot detection analysis
  4. 6-panel comparison visualization per test sample
  5. Summary report saved as results_report.txt
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, root_dir)

import torch
from src.models.unet import CongestionUNet
from src.metrics.eval_metrics import evaluate_all_metrics, compute_pcc, compute_nmse, compute_hotspot_metrics
from src.features.dataset import create_train_val_split


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> tuple:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})
    model = CongestionUNet(
        in_channels=cfg.get("data", {}).get("in_channels", 6),
        out_channels=cfg.get("data", {}).get("out_channels", 2),
        base_channels=cfg.get("model", {}).get("base_channels", 32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def predict(model, features_np: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(features_np).unsqueeze(0)
        return model(x).squeeze(0).numpy()


def compute_iou_hotspot(pred, target, top_k=0.05):
    """Intersection over Union for top-k% hotspot tiles."""
    p_flat, t_flat = pred.flatten(), target.flatten()
    k = max(1, int(len(t_flat) * top_k))
    top_p = set(np.argsort(p_flat)[-k:])
    top_t = set(np.argsort(t_flat)[-k:])
    inter = len(top_p & top_t)
    union = len(top_p | top_t)
    return inter / union if union > 0 else 0.0


def compute_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))


def compute_rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def pct_within(pred, target, tol=0.1):
    """% of pixels where |pred-target| < tol."""
    return float(np.mean(np.abs(pred - target) < tol) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample(features, target, pred, metrics, save_path, title=""):
    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    ch_names = ["Pin Density", "Cell Density", "RUDY-H", "RUDY-V", "RUDY Total", "Macro Mask"]
    cmaps_feat = ["plasma", "magma", "Blues", "Greens", "viridis", "gray"]

    # Row 0: 4 input features
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(features[i], cmap=cmaps_feat[i], origin="lower", aspect="auto")
        ax.set_title(f"Input: {ch_names[i]}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])

    # Row 1: remaining 2 inputs + targets
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(features[4], cmap="viridis", origin="lower", aspect="auto")
    ax.set_title("Input: RUDY Total", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(features[5], cmap="gray", origin="lower", aspect="auto")
    ax.set_title("Input: Macro Blockage", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    # H-overflow: GT vs Pred
    vmax_h = max(float(target[0].max()), float(pred[0].max()), 0.01)
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(target[0], cmap="hot", origin="lower", vmin=0, vmax=vmax_h, aspect="auto")
    ax.set_title("GT: H-Overflow", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(pred[0], cmap="hot", origin="lower", vmin=0, vmax=vmax_h, aspect="auto")
    h_pcc = metrics.get('H_Congestion_pcc', metrics.get('H_overflow_pcc', 0.0))
    ax.set_title(f"Pred: H-Congestion\nPCC={h_pcc:.3f}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    # V-overflow: GT vs Pred + Error
    vmax_v = max(float(target[1].max()), float(pred[1].max()), 0.01)
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(target[1], cmap="hot", origin="lower", vmin=0, vmax=vmax_v, aspect="auto")
    ax.set_title("GT: V-Congestion", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(pred[1], cmap="hot", origin="lower", vmin=0, vmax=vmax_v, aspect="auto")
    v_pcc = metrics.get('V_Congestion_pcc', metrics.get('V_overflow_pcc', 0.0))
    ax.set_title(f"Pred: V-Congestion\nPCC={v_pcc:.3f}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    # Error maps
    err_h = np.abs(pred[0] - target[0])
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(err_h, cmap="inferno", origin="lower", aspect="auto")
    ax.set_title(f"Error Map H\nMAE={compute_mae(pred[0],target[0]):.4f}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    err_v = np.abs(pred[1] - target[1])
    ax = fig.add_subplot(gs[2, 3])
    im = ax.imshow(err_v, cmap="inferno", origin="lower", aspect="auto")
    ax.set_title(f"Error Map V\nMAE={compute_mae(pred[1],target[1]):.4f}", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.set_xticks([]); ax.set_yticks([])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_summary(all_metrics, save_path):
    """Bar charts of all key metrics across test samples."""
    n = len(all_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Test Set Metrics — CircuitNet (nvdla-small)", fontsize=14, fontweight="bold")

    metric_defs = [
        ("mean_pcc",              "Mean PCC (↑ better)",        "steelblue",  [0, 1]),
        ("mean_nmse",             "Mean NMSE (↓ better)",       "tomato",     None),
        ("mean_hotspot_f1_top5pct","Hotspot F1 @Top-5% (↑)",   "seagreen",   [0, 1]),
        ("H_Congestion_pcc",      "H-Congestion PCC",           "orchid",     [0, 1]),
        ("V_Congestion_pcc",      "V-Congestion PCC",           "darkorange", [0, 1]),
        ("H_Congestion_nmse",     "H-Congestion NMSE (↓)",      "slategray",  None),
    ]

    for ax, (key, label, color, ylim) in zip(axes.flat, metric_defs):
        # Support fallback to overflow if Congestion not present
        alt_key = key.replace("Congestion", "overflow")
        vals = [m.get(key, m.get(alt_key, 0.0)) for m in all_metrics]
        bars = ax.bar(range(n), vals, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(np.mean(vals), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean = {np.mean(vals):.4f}")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel(key)
        ax.legend(fontsize=9)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)], fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Metrics chart saved → {save_path}")


def plot_loss_curve(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "o-", color="steelblue", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses,   "s-", color="tomato",    label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss Curve — CircuitNet", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Loss curve saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full test evaluation & results report")
    parser.add_argument("--checkpoint", type=str,
                        default="./outputs/checkpoints/best_congestion_unet.pt")
    parser.add_argument("--data_dir", type=str,
                        default="./data/circuitnet_processed")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/test_results")
    parser.add_argument("--max_vis", type=int, default=4,
                        help="Max number of sample visualizations to generate")
    args = parser.parse_args()

    ckpt_path = os.path.join(root_dir, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    data_dir  = os.path.join(root_dir, args.data_dir)   if not os.path.isabs(args.data_dir)   else args.data_dir
    out_dir   = os.path.join(root_dir, args.output_dir) if not os.path.isabs(args.output_dir)  else args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found: {ckpt_path}")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(" CONGESTION HEATMAP PREDICTION — FULL TEST EVALUATION")
    print(f"{'='*65}")
    model, cfg, ckpt = load_model(ckpt_path)
    print(f"[✓] Model loaded from: {ckpt_path}")
    print(f"    Trained for epoch {ckpt.get('epoch','?')}  |  "
          f"Best val F1@5%: {ckpt.get('hotspot_f1', 0):.4f}  |  "
          f"Data: {ckpt.get('data_source','?')}")

    # ── Load test files (use val split as held-out test set) ─────────────────
    _, test_files = create_train_val_split(data_dir, val_ratio=0.15, seed=42)
    if not test_files:
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        test_files = all_files[-8:]  # last 8 as test
    print(f"[*] Test samples: {len(test_files)}")

    # ── Run inference on every test sample ───────────────────────────────────
    all_metrics   = []
    all_preds     = []
    all_targets   = []
    all_features  = []
    inference_times = []

    print(f"\n{'Sample':<10} {'PCC':>8} {'NMSE':>8} {'MAE':>8} {'RMSE':>8} {'F1@5%':>8} {'IoU@5%':>8} {'Within10%':>10}")
    print("-" * 75)

    for i, fpath in enumerate(test_files):
        d = np.load(fpath)
        features = d["features"].astype(np.float32)
        target   = d["target"].astype(np.float32)

        t0 = time.time()
        pred = predict(model, features)
        inf_ms = (time.time() - t0) * 1000
        inference_times.append(inf_ms)

        m = evaluate_all_metrics(pred, target)
        # Extra metrics
        m["mae_h"]    = compute_mae(pred[0], target[0])
        m["mae_v"]    = compute_mae(pred[1], target[1])
        m["rmse_h"]   = compute_rmse(pred[0], target[0])
        m["rmse_v"]   = compute_rmse(pred[1], target[1])
        m["iou_top5"] = compute_iou_hotspot(pred[0], target[0], 0.05)
        m["pct_within_10pct"] = pct_within(pred[0], target[0], 0.1)
        m["inference_ms"] = inf_ms

        all_metrics.append(m)
        all_preds.append(pred)
        all_targets.append(target)
        all_features.append(features)

        name = os.path.basename(fpath).replace(".npz", "")
        print(f"{name[-12:]:<12} {m['mean_pcc']:>8.4f} {m['mean_nmse']:>8.4f} "
              f"{(m['mae_h']+m['mae_v'])/2:>8.4f} "
              f"{(m['rmse_h']+m['rmse_v'])/2:>8.4f} "
              f"{m['mean_hotspot_f1_top5pct']:>8.4f} "
              f"{m['iou_top5']:>8.4f} "
              f"{m['pct_within_10pct']:>10.1f}%")

    # ── Aggregate Summary ─────────────────────────────────────────────────────
    def avg(key): return float(np.mean([m[key] for m in all_metrics]))
    def std(key): return float(np.std([m[key]  for m in all_metrics]))

    print(f"\n{'='*65}")
    print(" SUMMARY RESULTS")
    print(f"{'='*65}")
    summary_lines = [
        ("Test samples",          len(test_files)),
        ("Avg inference time",    f"{avg('inference_ms'):.1f} ms/sample"),
        ("", ""),
        ("── Regression Metrics ──────────────────────────", ""),
        ("Mean PCC  (↑ better)",  f"{avg('mean_pcc'):.4f}  ± {std('mean_pcc'):.4f}"),
        ("Mean NMSE (↓ better)",  f"{avg('mean_nmse'):.4f}  ± {std('mean_nmse'):.4f}"),
        ("Mean MAE  (↓ better)",  f"{avg('mae_h'):.5f}"),
        ("Mean RMSE (↓ better)",  f"{avg('rmse_h'):.5f}"),
        ("", ""),
        ("── Hotspot Detection (Critical for EDA) ────────", ""),
        ("Hotspot F1  @Top-5%",   f"{avg('mean_hotspot_f1_top5pct'):.4f}  ± {std('mean_hotspot_f1_top5pct'):.4f}"),
        ("Hotspot IoU @Top-5%",   f"{avg('iou_top5'):.4f}  ± {std('iou_top5'):.4f}"),
        ("Pixels within ±10%",    f"{avg('pct_within_10pct'):.1f}%"),
        ("", ""),
        ("── Per-Channel Breakdown ───────────────────────", ""),
        ("H-Congestion PCC",      f"{avg('H_Congestion_pcc') if 'H_Congestion_pcc' in all_metrics[0] else avg('H_overflow_pcc'):.4f}"),
        ("V-Congestion PCC",      f"{avg('V_Congestion_pcc') if 'V_Congestion_pcc' in all_metrics[0] else avg('V_overflow_pcc'):.4f}"),
        ("H-Congestion NMSE",     f"{avg('H_Congestion_nmse') if 'H_Congestion_nmse' in all_metrics[0] else avg('H_overflow_nmse'):.4f}"),
        ("V-Congestion NMSE",     f"{avg('V_Congestion_nmse') if 'V_Congestion_nmse' in all_metrics[0] else avg('V_overflow_nmse'):.4f}"),
        ("H-Congestion F1@5%",    f"{avg('H_Congestion_hotspot_top_5pct_f1') if 'H_Congestion_hotspot_top_5pct_f1' in all_metrics[0] else avg('H_overflow_hotspot_top_5pct_f1'):.4f}"),
        ("V-Congestion F1@5%",    f"{avg('V_Congestion_hotspot_top_5pct_f1') if 'V_Congestion_hotspot_top_5pct_f1' in all_metrics[0] else avg('V_overflow_hotspot_top_5pct_f1'):.4f}"),
        ("H-Congestion F1@10%",   f"{avg('H_Congestion_hotspot_top_10pct_f1') if 'H_Congestion_hotspot_top_10pct_f1' in all_metrics[0] else avg('H_overflow_hotspot_top_10pct_f1'):.4f}"),
        ("V-Congestion F1@10%",   f"{avg('V_Congestion_hotspot_top_10pct_f1') if 'V_Congestion_hotspot_top_10pct_f1' in all_metrics[0] else avg('V_overflow_hotspot_top_10pct_f1'):.4f}"),
    ]

    for label, value in summary_lines:
        if label == "":
            print()
        elif str(label).startswith("──"):
            print(f"\n{label}")
        else:
            print(f"  {label:<40} {value}")

    print(f"\n{'='*65}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "results_report.txt")
    with open(report_path, "w") as f:
        f.write("CONGESTION HEATMAP PREDICTION — TEST RESULTS\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Data: {data_dir}\n")
        f.write(f"Test samples: {len(test_files)}\n\n")
        for label, value in summary_lines:
            if str(label).startswith("──"):
                f.write(f"\n{label}\n")
            elif label == "":
                f.write("\n")
            else:
                f.write(f"  {label:<40} {value}\n")

        f.write("\n\nPer-Sample Details:\n")
        f.write(f"{'File':<35} {'PCC':>8} {'NMSE':>8} {'F1@5%':>8} {'IoU@5%':>8}\n")
        f.write("-" * 70 + "\n")
        for fpath, m in zip(test_files, all_metrics):
            name = os.path.basename(fpath)
            f.write(f"{name:<35} {m['mean_pcc']:>8.4f} {m['mean_nmse']:>8.4f} "
                    f"{m['mean_hotspot_f1_top5pct']:>8.4f} {m['iou_top5']:>8.4f}\n")

    print(f"[✓] Report saved → {report_path}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print(f"\n[*] Generating {min(args.max_vis, len(test_files))} sample visualizations...")
    for i in range(min(args.max_vis, len(test_files))):
        name = os.path.basename(test_files[i]).replace(".npz", "")
        vis_path = os.path.join(out_dir, f"sample_{i:02d}_{name}.png")
        plot_sample(
            all_features[i], all_targets[i], all_preds[i], all_metrics[i],
            vis_path,
            title=f"Test Sample {i}: {name}\n"
                  f"PCC={all_metrics[i]['mean_pcc']:.4f}  "
                  f"NMSE={all_metrics[i]['mean_nmse']:.4f}  "
                  f"F1@5%={all_metrics[i]['mean_hotspot_f1_top5pct']:.4f}  "
                  f"IoU={all_metrics[i]['iou_top5']:.4f}"
        )
        print(f"  [✓] sample_{i:02d}.png saved")

    # ── Metrics bar chart ─────────────────────────────────────────────────────
    plot_metrics_summary(all_metrics, os.path.join(out_dir, "metrics_summary.png"))

    print(f"\n[✓] All results saved to: {out_dir}")
    print(f"[✓] Files generated:")
    for f in sorted(os.listdir(out_dir)):
        fsize = os.path.getsize(os.path.join(out_dir, f))
        print(f"    {f}  ({fsize/1024:.0f} KB)")


if __name__ == "__main__":
    main()
