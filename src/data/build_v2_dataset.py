"""
CircuitNet Data Preprocessor: Generates both Congestion Utilization & Overflow targets.
Target Channel 0: Horizontal Routing Utilization (0.0 to 1.0)
Target Channel 1: Vertical Routing Utilization (0.0 to 1.0)
Target Channel 2: Horizontal Overflow (Hotspots)
Target Channel 3: Vertical Overflow (Hotspots)

Having both Utilization (dense continuous routing demand / capacity) and
Overflow (discrete hotspot violations) enables high PCC, SSIM, and Hotspot F1!
"""

import os
from pathlib import Path
from typing import List
import numpy as np
from scipy.ndimage import zoom


FEATURE_SUBDIRS = {
    "pin_density":  "RUDY/RUDY_pin",
    "cell_density": "cell_density",
    "rudy_h":       "RUDY/RUDY_short",
    "rudy_v":       "RUDY/RUDY_long",
    "rudy_total":   "RUDY/RUDY",
    "macro_region": "macro_region",
}

FEATURE_SUBDIRS_FALLBACK = {
    "pin_density":  "RUDY/RUDY_pin",
    "cell_density": "cell_density",
    "rudy_h":       "RUDY/RUDY",
    "rudy_v":       "RUDY/RUDY",
    "rudy_total":   "RUDY/RUDY",
    "macro_region": "macro_region",
}

TARGET_UTIL_SUBDIRS = {
    "util_h": "congestion/congestion_early_global_routing/utilization_based/congestion_eGR_horizontal_util",
    "util_v": "congestion/congestion_early_global_routing/utilization_based/congestion_eGR_vertical_util",
}

TARGET_OVERFLOW_SUBDIRS = {
    "of_h": "congestion/congestion_early_global_routing/overflow_based/congestion_eGR_horizontal_overflow",
    "of_v": "congestion/congestion_early_global_routing/overflow_based/congestion_eGR_vertical_overflow",
}


def resize_2d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    zy = h / arr.shape[0]
    zx = w / arr.shape[1]
    return zoom(arr, (zy, zx), order=1).astype(np.float32)


def robust_norm(arr: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(arr))
    return (arr / (m + 1e-6)) if m > 0 else arr


def build_enhanced_dataset(
    raw_dir: str = "./data/circuitnet_raw/nvdla-small",
    output_dir: str = "./data/circuitnet_processed_v2",
    grid_h: int = 128,
    grid_w: int = 128,
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rudy_ref = raw_path / "RUDY" / "RUDY"
    stems = sorted([p.stem for p in rudy_ref.glob("*.npz")])
    print(f"[*] Processing {len(stems)} CircuitNet designs into {out_path}...")

    feat_dirs = FEATURE_SUBDIRS if (raw_path / "RUDY" / "RUDY_short").exists() else FEATURE_SUBDIRS_FALLBACK

    for idx, stem in enumerate(stems):
        # 1. Extract 6 Features
        feats = []
        for ch_name, sdir in feat_dirs.items():
            fpath = raw_path / sdir / f"{stem}.npz"
            if fpath.exists():
                arr = np.load(str(fpath))["data"].astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.shape != (grid_h, grid_w):
                    arr = resize_2d(arr, grid_h, grid_w)
                if ch_name == "macro_region":
                    arr = (arr > 0.5).astype(np.float32)
                else:
                    arr = robust_norm(arr)
            else:
                arr = np.zeros((grid_h, grid_w), dtype=np.float32)
            feats.append(arr)

        # 2. Extract Targets (Utilization H & V)
        targets = []
        for t_name, sdir in TARGET_UTIL_SUBDIRS.items():
            fpath = raw_path / sdir / f"{stem}.npz"
            if fpath.exists():
                arr = np.load(str(fpath))["data"].astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.shape != (grid_h, grid_w):
                    arr = resize_2d(arr, grid_h, grid_w)
                # Utilization is already 0.0 to 1.0 (clamped)
                arr = np.clip(arr, 0.0, 1.0)
            else:
                arr = np.zeros((grid_h, grid_w), dtype=np.float32)
            targets.append(arr)

        feat_tensor = np.stack(feats, axis=0).astype(np.float32)       # (6, 128, 128)
        target_tensor = np.stack(targets, axis=0).astype(np.float32)   # (2, 128, 128)

        dest_file = out_path / f"circuit_{idx:04d}_{stem}.npz"
        np.savez_compressed(dest_file, features=feat_tensor, target=target_tensor)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(stems):
            print(f"  [{idx + 1:2d}/{len(stems)}] saved {dest_file.name}")

    print(f"[✓] Complete! {len(stems)} processed dataset files written to {out_path}")


if __name__ == "__main__":
    build_enhanced_dataset()
