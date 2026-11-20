"""
Lightweight parser for standard DEF (Design Exchange Format) files.
Extracts component placement, pin coordinates, and net connectivity for EDA ML features.
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class DefParser:
    """
    Parses standard DEF files to extract:
    - Die area bounding box
    - Standard cell & macro placements (x, y, orient, type)
    - Net bounding boxes and pin connections
    """

    def __init__(self, def_file_path: str):
        self.def_file_path = def_file_path
        self.die_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0)
        self.components: Dict[str, Dict[str, Any]] = {}
        self.nets: Dict[str, List[Tuple[float, float]]] = {}

    def parse(self) -> Dict[str, Any]:
        """Parses the DEF file and returns structured layout information."""
        with open(self.def_file_path, "r", errors="ignore") as f:
            lines = f.readlines()

        in_components = False
        in_nets = False
        current_net = None

        die_pattern = re.compile(r"DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)")
        comp_pattern = re.compile(r"-\s+(\S+)\s+(\S+).*\(\s*(\d+)\s+(\d+)\s*\)\s+(\S+)")

        for line in lines:
            line_str = line.strip()

            # 1. Die Area
            if line_str.startswith("DIEAREA"):
                m = die_pattern.search(line_str)
                if m:
                    self.die_bbox = (float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))

            # 2. Components
            elif line_str.startswith("COMPONENTS"):
                in_components = True
            elif in_components and line_str.startswith("END COMPONENTS"):
                in_components = False
            elif in_components and line_str.startswith("-"):
                m = comp_pattern.search(line_str)
                if m:
                    comp_name, cell_type, x, y, orient = m.groups()
                    self.components[comp_name] = {
                        "cell_type": cell_type,
                        "x": float(x),
                        "y": float(y),
                        "orient": orient,
                    }

            # 3. Nets
            elif line_str.startswith("NETS"):
                in_nets = True
            elif in_nets and line_str.startswith("END NETS"):
                in_nets = False
            elif in_nets and line_str.startswith("-"):
                parts = line_str.split()
                if len(parts) >= 2:
                    current_net = parts[1]
                    self.nets[current_net] = []
            elif in_nets and current_net:
                # Extract connected component names in ( comp_name pin_name )
                pin_matches = re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", line_str)
                for comp_name, pin_name in pin_matches:
                    if comp_name in self.components:
                        # Approximate pin location with cell position if pin offset is not defined
                        cx = self.components[comp_name]["x"]
                        cy = self.components[comp_name]["y"]
                        self.nets[current_net].append((cx, cy))

        return {
            "die_bbox": self.die_bbox,
            "components": self.components,
            "nets": self.nets,
        }

    def get_net_bboxes(self) -> np.ndarray:
        """Computes bounding boxes [x1, y1, x2, y2] for all multi-pin nets."""
        bboxes = []
        for net_name, pins in self.nets.items():
            if len(pins) >= 2:
                coords = np.array(pins)
                x1, y1 = np.min(coords, axis=0)
                x2, y2 = np.max(coords, axis=0)
                bboxes.append([x1, y1, x2, y2])
        return np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32)
