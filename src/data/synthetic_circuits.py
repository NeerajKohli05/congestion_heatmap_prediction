"""
Synthetic VLSI Circuit Placement and Congestion Generator.
Emulates Rent's rule netlist distribution, macro placement, and global routing congestion.
Used for immediate testing, validation, and benchmarking before plugging in massive DEF/LEF files.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from ..features.rudy import RudyFeatureExtractor


class SyntheticCircuitGenerator:
    """
    Generates realistic synthetic circuit layout data:
    - Standard cell locations & bounding boxes
    - Macros / blockage regions
    - Nets following Rent's rule / log-normal wirelengths
    - Ground-truth routing congestion heatmaps (Horizontal & Vertical overflow)
    """

    def __init__(
        self,
        grid_h: int = 128,
        grid_w: int = 128,
        die_size: Tuple[float, float] = (1000.0, 1000.0),
        random_seed: int = 42
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.die_w, self.die_h = die_size
        self.die_bbox = (0.0, 0.0, self.die_w, self.die_h)
        self.rng = np.random.RandomState(random_seed)
        self.feature_extractor = RudyFeatureExtractor(grid_h, grid_w, self.die_bbox)

    def generate_single_design(
        self,
        num_cells: int = 1500,
        num_nets: int = 2000,
        num_macros: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        Generates one synthetic circuit design with placement, nets, and congestion maps.
        Returns:
            Dict containing:
                - 'features': (6, grid_h, grid_w) input feature tensor
                - 'target': (2, grid_h, grid_w) ground truth congestion (0: H-overflow, 1: V-overflow)
        """
        # 1. Place 2-6 hard macros (corner/edge locations typical of physical design)
        macro_boxes = []
        for _ in range(num_macros):
            mw = self.rng.uniform(100.0, 250.0)
            mh = self.rng.uniform(100.0, 250.0)
            # Bias macros towards chip boundaries / corners
            corner_choice = self.rng.choice([0, 1, 2, 3])
            if corner_choice == 0:
                mx = self.rng.uniform(0.0, 100.0)
                my = self.rng.uniform(0.0, 100.0)
            elif corner_choice == 1:
                mx = self.rng.uniform(self.die_w - mw - 100.0, self.die_w - mw)
                my = self.rng.uniform(0.0, 100.0)
            elif corner_choice == 2:
                mx = self.rng.uniform(0.0, 100.0)
                my = self.rng.uniform(self.die_h - mh - 100.0, self.die_h - mh)
            else:
                mx = self.rng.uniform(self.die_w - mw - 100.0, self.die_w - mw)
                my = self.rng.uniform(self.die_h - mh - 100.0, self.die_h - mh)

            macro_boxes.append([mx, my, mx + mw, my + mh])
        macro_boxes = np.array(macro_boxes, dtype=np.float32) if macro_boxes else np.zeros((0, 4), dtype=np.float32)

        # 2. Place standard cells in clustered logic modules (hierarchical layout structure)
        num_clusters = self.rng.randint(4, 10)
        cluster_centers = self.rng.uniform(0.15 * self.die_w, 0.85 * self.die_w, size=(num_clusters, 2))

        cell_boxes = []
        cell_centers = []
        for _ in range(num_cells):
            cluster_idx = self.rng.choice(num_clusters)
            center = cluster_centers[cluster_idx]
            cx = self.rng.normal(center[0], 0.08 * self.die_w)
            cy = self.rng.normal(center[1], 0.08 * self.die_h)
            cx = np.clip(cx, 10.0, self.die_w - 10.0)
            cy = np.clip(cy, 10.0, self.die_h - 10.0)

            cw = self.rng.uniform(4.0, 12.0)
            ch = self.rng.uniform(4.0, 8.0)
            cell_boxes.append([cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2])
            cell_centers.append([cx, cy])

        cell_boxes = np.array(cell_boxes, dtype=np.float32)
        cell_centers = np.array(cell_centers, dtype=np.float32)

        # 3. Generate pins (approx 3 pins per cell)
        pin_coords = []
        for c in cell_centers:
            for _ in range(self.rng.randint(2, 5)):
                px = c[0] + self.rng.uniform(-3.0, 3.0)
                py = c[1] + self.rng.uniform(-3.0, 3.0)
                pin_coords.append([px, py])
        pin_coords = np.array(pin_coords, dtype=np.float32)

        # 4. Generate Nets (Rent's rule: mostly local short nets, some long global nets)
        net_bboxes = []
        demand_h = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        demand_v = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        for _ in range(num_nets):
            # 85% local nets, 15% long inter-cluster nets
            is_global = self.rng.rand() < 0.15
            pin_count = self.rng.choice([2, 3, 4, 5], p=[0.65, 0.20, 0.10, 0.05])

            if is_global:
                chosen_pins = pin_coords[self.rng.choice(len(pin_coords), pin_count, replace=False)]
            else:
                pivot_idx = self.rng.randint(0, len(pin_coords))
                pivot = pin_coords[pivot_idx]
                dists = np.sum((pin_coords - pivot) ** 2, axis=1)
                near_candidates = np.argsort(dists)[:80]
                chosen_pins = pin_coords[self.rng.choice(near_candidates, pin_count, replace=False)]

            x1, y1 = np.min(chosen_pins, axis=0)
            x2, y2 = np.max(chosen_pins, axis=0)
            net_bboxes.append([x1, y1, x2, y2])

            # Routing demand emulation: L-shaped and probabilistic routing paths
            c1 = np.clip(int(x1 / self.die_w * self.grid_w), 0, self.grid_w - 1)
            c2 = np.clip(int(x2 / self.die_w * self.grid_w), 0, self.grid_w - 1)
            r1 = np.clip(int(y1 / self.die_h * self.grid_h), 0, self.grid_h - 1)
            r2 = np.clip(int(y2 / self.die_h * self.grid_h), 0, self.grid_h - 1)

            # Route along grid tracks
            demand_h[r1, min(c1, c2) : max(c1, c2) + 1] += 1.0
            demand_v[min(r1, r2) : max(r1, r2) + 1, c2] += 1.0

        net_bboxes = np.array(net_bboxes, dtype=np.float32)

        # 5. Routing Capacity & Overflow computation
        # Capacity is lower near macros due to blockages and routing pin constraints
        base_cap_h = 14.0
        base_cap_v = 14.0
        capacity_h = np.ones((self.grid_h, self.grid_w), dtype=np.float32) * base_cap_h
        capacity_v = np.ones((self.grid_h, self.grid_w), dtype=np.float32) * base_cap_v

        # Reduce capacity where macros exist
        for mbox in macro_boxes:
            c1 = np.clip(int(mbox[0] / self.die_w * self.grid_w), 0, self.grid_w - 1)
            c2 = np.clip(int(mbox[2] / self.die_w * self.grid_w), 0, self.grid_w - 1)
            r1 = np.clip(int(mbox[1] / self.die_h * self.grid_h), 0, self.grid_h - 1)
            r2 = np.clip(int(mbox[3] / self.die_h * self.grid_h), 0, self.grid_h - 1)
            capacity_h[r1 : r2 + 1, c1 : c2 + 1] *= 0.25
            capacity_v[r1 : r2 + 1, c1 : c2 + 1] *= 0.25

        # Compute Ground Truth Congestion: Overflow = max(0, Demand - Capacity)
        overflow_h = np.maximum(0.0, demand_h - capacity_h)
        overflow_v = np.maximum(0.0, demand_v - capacity_v)

        # Normalize targets so they scale smoothly (ratio: demand / capacity)
        norm_cong_h = demand_h / (capacity_h + 1e-4)
        norm_cong_v = demand_v / (capacity_v + 1e-4)

        # Extract features via RudyFeatureExtractor
        features = self.feature_extractor.extract_all_channels(
            pin_coords=pin_coords,
            cell_boxes=cell_boxes,
            net_bboxes=net_bboxes,
            macro_boxes=macro_boxes,
            die_bbox=self.die_bbox
        )

        target = np.stack([norm_cong_h, norm_cong_v], axis=0)  # Shape: (2, H, W)

        return {
            "features": features.astype(np.float32),
            "target": target.astype(np.float32)
        }

    def generate_dataset_split(
        self,
        num_samples: int = 50,
        output_dir: str = "./data/processed"
    ) -> List[str]:
        """Generates and saves multiple designs as compressed .npz files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i in range(num_samples):
            # Vary cell count and nets across samples for diversity
            cells = self.rng.randint(1200, 2200)
            nets = self.rng.randint(1600, 2800)
            sample = self.generate_single_design(num_cells=cells, num_nets=nets)

            filepath = os.path.join(output_dir, f"design_{i:04d}.npz")
            np.savez_compressed(
                filepath,
                features=sample["features"],
                target=sample["target"]
            )
            saved_paths.append(filepath)

        return saved_paths
