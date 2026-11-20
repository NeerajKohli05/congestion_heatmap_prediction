"""
PyTorch Dataset and DataLoader for Circuit Congestion Heatmaps.
Supports data augmentation (flips, 90-degree rotations preserving physical symmetry).
"""

import os
import glob
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class CircuitCongestionDataset(Dataset):
    """
    PyTorch Dataset loader for circuit layout feature grids and congestion targets.
    Reads .npz files containing:
      - 'features': (C_in, H, W)
      - 'target': (C_out, H, W)
    """

    def __init__(
        self,
        file_paths: List[str],
        augment: bool = False,
        normalize_features: bool = True
    ):
        self.file_paths = file_paths
        self.augment = augment
        self.normalize_features = normalize_features

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.file_paths[idx])
        features = data["features"].astype(np.float32)  # (C, H, W)
        target = data["target"].astype(np.float32)      # (2, H, W)

        if self.augment:
            features, target = self._apply_augmentation(features, target)

        return torch.from_numpy(features), torch.from_numpy(target)

    def _apply_augmentation(
        self,
        features: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Spatially consistent EDA augmentation.
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=2).copy()
            target = np.flip(target, axis=2).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=1).copy()
            target = np.flip(target, axis=1).copy()

        # Random 180-degree rotation
        if np.random.rand() > 0.5:
            features = np.rot90(features, 2, axes=(1, 2)).copy()
            target = np.rot90(target, 2, axes=(1, 2)).copy()

        return features, target


def create_train_val_split(
    data_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Splits dataset files into deterministic train and validation sets."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(files))
    val_count = max(1, int(len(files) * val_ratio))

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_files = [files[i] for i in train_indices]
    val_files = [files[i] for i in val_indices]

    return train_files, val_files
