"""
Evaluation Metrics for Circuit Routing Congestion Heatmap Prediction.
Includes NMSE, Pearson Correlation Coefficient (PCC), SSIM, MAE, RMSE,
and Hotspot Top-k Precision, Recall, F1, and IoU (Intersection-over-Union).
"""

from typing import Dict
import numpy as np


def compute_nmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalized Mean Squared Error: ||pred - target||^2 / ||target||^2."""
    mse = np.mean((pred - target) ** 2)
    norm = np.mean(target ** 2) + 1e-8
    return float(mse / norm)


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_pcc(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Pearson Correlation Coefficient (PCC) across all tiles.
    Range: [-1.0, 1.0], higher is better.
    """
    p_flat = pred.flatten()
    t_flat = target.flatten()

    p_diff = p_flat - np.mean(p_flat)
    t_diff = t_flat - np.mean(t_flat)

    numerator = np.sum(p_diff * t_diff)
    denominator = np.sqrt(np.sum(p_diff ** 2) * np.sum(t_diff ** 2)) + 1e-8
    return float(numerator / denominator)


def compute_hotspot_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    top_k_percent: float = 0.05
) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1-score, and IoU for the top-k% most congested tiles.
    Args:
        pred: Predicted congestion map.
        target: Ground truth congestion map.
        top_k_percent: Fraction of tiles considered as true hotspots (e.g. 0.05 = top 5%).
    """
    p_flat = pred.flatten()
    t_flat = target.flatten()

    k = max(1, int(len(t_flat) * top_k_percent))

    top_t_indices = set(np.argsort(t_flat)[-k:])
    top_p_indices = set(np.argsort(p_flat)[-k:])

    intersection = len(top_t_indices.intersection(top_p_indices))
    union = len(top_t_indices.union(top_p_indices))

    precision = intersection / k
    recall = intersection / k
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = intersection / (union + 1e-8)

    pct_str = f"{int(top_k_percent * 100)}pct"
    return {
        f"hotspot_top_{pct_str}_precision": float(precision),
        f"hotspot_top_{pct_str}_recall": float(recall),
        f"hotspot_top_{pct_str}_f1": float(f1),
        f"hotspot_top_{pct_str}_iou": float(iou),
    }


def compute_ssim_np(pred: np.ndarray, target: np.ndarray) -> float:
    """Fast NumPy calculation of Structural Similarity Index (SSIM)."""
    mu_x = np.mean(pred)
    mu_y = np.mean(target)
    sigma_x_sq = np.var(pred)
    sigma_y_sq = np.var(target)
    sigma_xy = np.mean((pred - mu_x) * (target - mu_y))

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-8
    )
    return float(ssim)


def evaluate_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Computes full suite of benchmark metrics across all output channels."""
    results = {}

    num_channels = pred.shape[0] if pred.ndim == 3 else 1
    channel_names = ["H_Congestion", "V_Congestion"] if num_channels == 2 else [f"ch_{i}" for i in range(num_channels)]

    for i in range(num_channels):
        p_c = pred[i] if pred.ndim == 3 else pred
        t_c = target[i] if target.ndim == 3 else target
        name = channel_names[i]

        results[f"{name}_nmse"] = compute_nmse(p_c, t_c)
        results[f"{name}_pcc"] = compute_pcc(p_c, t_c)
        results[f"{name}_mae"] = compute_mae(p_c, t_c)
        results[f"{name}_rmse"] = compute_rmse(p_c, t_c)
        results[f"{name}_ssim"] = compute_ssim_np(p_c, t_c)

        results.update({f"{name}_{k}": v for k, v in compute_hotspot_metrics(p_c, t_c, 0.05).items()})
        results.update({f"{name}_{k}": v for k, v in compute_hotspot_metrics(p_c, t_c, 0.10).items()})

    results["mean_pcc"] = float(np.mean([results[f"{n}_pcc"] for n in channel_names]))
    results["mean_nmse"] = float(np.mean([results[f"{n}_nmse"] for n in channel_names]))
    results["mean_mae"] = float(np.mean([results[f"{n}_mae"] for n in channel_names]))
    results["mean_rmse"] = float(np.mean([results[f"{n}_rmse"] for n in channel_names]))
    results["mean_ssim"] = float(np.mean([results[f"{n}_ssim"] for n in channel_names]))
    results["mean_hotspot_f1_top5pct"] = float(np.mean([results[f"{n}_hotspot_top_5pct_f1"] for n in channel_names]))
    results["mean_hotspot_f1_top10pct"] = float(np.mean([results[f"{n}_hotspot_top_10pct_f1"] for n in channel_names]))
    results["mean_hotspot_iou_top10pct"] = float(np.mean([results[f"{n}_hotspot_top_10pct_iou"] for n in channel_names]))

    return results
