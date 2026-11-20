"""
RUDY (Rectangular Uniform wire DensitY) and Spatial Feature Generator for Circuit Routing Congestion.
Computes multi-channel feature grids from cell placements and netlists.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class RudyFeatureExtractor:
    """
    Extracts spatial EDA feature maps for routing congestion prediction:
    - Channel 0: Pin Density Map
    - Channel 1: Cell Area Density Map
    - Channel 2: Horizontal RUDY (Wire demand estimation in horizontal direction)
    - Channel 3: Vertical RUDY (Wire demand estimation in vertical direction)
    - Channel 4: Total RUDY
    - Channel 5: Macro Blockage Mask
    """

    def __init__(self, grid_h: int = 128, grid_w: int = 128, die_bbox: Optional[Tuple[float, float, float, float]] = None):
        """
        Args:
            grid_h: Number of grid bins vertically (GCell rows).
            grid_w: Number of grid bins horizontally (GCell columns).
            die_bbox: (x_min, y_min, x_max, y_max) bounding box of the chip die.
        """
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.die_bbox = die_bbox

    def _coord_to_bin(self, x: float, y: float, die_bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
        """Maps physical layout coordinates to grid bin (row, col)."""
        x_min, y_min, x_max, y_max = die_bbox
        w = max(x_max - x_min, 1e-6)
        h = max(y_max - y_min, 1e-6)

        col = int((x - x_min) / w * self.grid_w)
        row = int((y - y_min) / h * self.grid_h)

        col = np.clip(col, 0, self.grid_w - 1)
        row = np.clip(row, 0, self.grid_h - 1)
        return row, col

    def compute_pin_density(
        self,
        pin_coords: np.ndarray,
        die_bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Computes 2D pin count density.
        Args:
            pin_coords: (N, 2) array of (x, y) pin coordinates.
            die_bbox: (x_min, y_min, x_max, y_max)
        Returns:
            (grid_h, grid_w) array of normalized pin density.
        """
        pin_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        if len(pin_coords) == 0:
            return pin_map

        x_min, y_min, x_max, y_max = die_bbox
        w = max(x_max - x_min, 1e-6)
        h = max(y_max - y_min, 1e-6)

        cols = np.clip(((pin_coords[:, 0] - x_min) / w * self.grid_w).astype(np.int32), 0, self.grid_w - 1)
        rows = np.clip(((pin_coords[:, 1] - y_min) / h * self.grid_h).astype(np.int32), 0, self.grid_h - 1)

        np.add.at(pin_map, (rows, cols), 1.0)
        return pin_map

    def compute_cell_density(
        self,
        cell_boxes: np.ndarray,
        die_bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Computes cell area density per tile.
        Args:
            cell_boxes: (N, 4) array of (x1, y1, x2, y2)
        Returns:
            (grid_h, grid_w) area density map
        """
        cell_map = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        if len(cell_boxes) == 0:
            return cell_map

        x_min, y_min, x_max, y_max = die_bbox
        w = max(x_max - x_min, 1e-6)
        h = max(y_max - y_min, 1e-6)

        for box in cell_boxes:
            bx1, by1, bx2, by2 = box
            area = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
            c1 = np.clip(int((bx1 - x_min) / w * self.grid_w), 0, self.grid_w - 1)
            c2 = np.clip(int((bx2 - x_min) / w * self.grid_w), 0, self.grid_w - 1)
            r1 = np.clip(int((by1 - y_min) / h * self.grid_h), 0, self.grid_h - 1)
            r2 = np.clip(int((by2 - y_min) / h * self.grid_h), 0, self.grid_h - 1)

            num_tiles = max((r2 - r1 + 1) * (c2 - c1 + 1), 1)
            distributed_area = area / num_tiles
            cell_map[r1 : r2 + 1, c1 : c2 + 1] += distributed_area

        return cell_map

    def compute_rudy(
        self,
        net_bboxes: np.ndarray,
        die_bbox: Tuple[float, float, float, float],
        net_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Horizontal, Vertical, and Total RUDY maps.
        Args:
            net_bboxes: (M, 4) array of (x1, y1, x2, y2) bounding box for each net.
            die_bbox: (x_min, y_min, x_max, y_max)
            net_weights: (M,) array of optional net weights (default 1.0)
        Returns:
            (rudy_h, rudy_v, rudy_total) each shape (grid_h, grid_w)
        """
        rudy_h = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        rudy_v = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        if len(net_bboxes) == 0:
            return rudy_h, rudy_v, rudy_h + rudy_v

        x_min, y_min, x_max, y_max = die_bbox
        w_die = max(x_max - x_min, 1e-6)
        h_die = max(y_max - y_min, 1e-6)

        bin_w = w_die / self.grid_w
        bin_h = h_die / self.grid_h

        if net_weights is None:
            net_weights = np.ones(len(net_bboxes), dtype=np.float32)

        for i, box in enumerate(net_bboxes):
            x1, y1, x2, y2 = box
            weight = net_weights[i]

            dx = max(x2 - x1, bin_w)
            dy = max(y2 - y1, bin_h)
            bbox_area = dx * dy

            # Bounding box in grid index space
            c1 = np.clip(int((x1 - x_min) / w_die * self.grid_w), 0, self.grid_w - 1)
            c2 = np.clip(int((x2 - x_min) / w_die * self.grid_w), 0, self.grid_w - 1)
            r1 = np.clip(int((y1 - y_min) / h_die * self.grid_h), 0, self.grid_h - 1)
            r2 = np.clip(int((y2 - y_min) / h_die * self.grid_h), 0, self.grid_h - 1)

            h_demand = (dx / bbox_area) * weight
            v_demand = (dy / bbox_area) * weight

            rudy_h[r1 : r2 + 1, c1 : c2 + 1] += h_demand
            rudy_v[r1 : r2 + 1, c1 : c2 + 1] += v_demand

        rudy_total = rudy_h + rudy_v
        return rudy_h, rudy_v, rudy_total

    def extract_all_channels(
        self,
        pin_coords: np.ndarray,
        cell_boxes: np.ndarray,
        net_bboxes: np.ndarray,
        macro_boxes: Optional[np.ndarray] = None,
        die_bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> np.ndarray:
        """
        Generates unified 6-channel feature map tensor:
        (Channels, Grid_H, Grid_W)
        """
        bbox = die_bbox or self.die_bbox
        if bbox is None:
            raise ValueError("die_bbox must be provided")

        pin_map = self.compute_pin_density(pin_coords, bbox)
        cell_map = self.compute_cell_density(cell_boxes, bbox)
        rudy_h, rudy_v, rudy_total = self.compute_rudy(net_bboxes, bbox)

        if macro_boxes is not None and len(macro_boxes) > 0:
            macro_map = self.compute_cell_density(macro_boxes, bbox)
            macro_mask = (macro_map > 0).astype(np.float32)
        else:
            macro_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Normalize individual channels gracefully
        def robust_norm(arr: np.ndarray) -> np.ndarray:
            max_v = np.max(arr)
            return (arr / (max_v + 1e-6)) if max_v > 0 else arr

        features = np.stack([
            robust_norm(pin_map),
            robust_norm(cell_map),
            robust_norm(rudy_h),
            robust_norm(rudy_v),
            robust_norm(rudy_total),
            macro_mask
        ], axis=0)  # Shape: (6, H, W)

        return features
