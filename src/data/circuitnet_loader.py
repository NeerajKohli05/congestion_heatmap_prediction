"""
CircuitNet Data Loader & Preprocessor.
Downloads routability feature archives from HuggingFace (CircuitNet/CircuitNet),
extracts them, and converts the per-variant .npz feature maps into our standard
6-channel .npz format compatible with the existing training pipeline.

REAL CircuitNet N14 routability feature folder structure (after extraction):
  nvdla-small/
    RUDY/
      RUDY/           -> total RUDY  shape (809, 802) key='data'
      RUDY_pin/       -> pin  RUDY   shape (809, 802) key='data'
      RUDY_long/      -> long-net RUDY
      RUDY_short/     -> short-net RUDY
    cell_density/     -> cell area density   shape (809, 802) key='data'
    macro_region/     -> binary blockage     shape (809, 802) key='data'
    congestion/
      congestion_early_global_routing/
        overflow_based/
          congestion_eGR_horizontal_overflow/ shape (1462, 1461) key='data'
          congestion_eGR_vertical_overflow/   shape (1462, 1461) key='data'

Each sub-folder contains 96 .npz files (one per placement variant).
All files in all folders share the SAME stem, e.g.:
    nvdla-small_freq_500_mp_1_fpu_55_fpa_1.0_p_1_fi_ap.npz
"""

import os
import tarfile
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ── Exact folder paths inside the extracted design root ───────────────────────
# Features are (809, 802) arrays, stored under key='data'
FEATURE_SUBDIRS = {
    "pin_density":  "RUDY/RUDY_pin",    # pin-based RUDY estimate
    "cell_density": "cell_density",      # standard-cell area density
    "rudy_h":       "RUDY/RUDY_short",  # short nets → proxy for H demand
    "rudy_v":       "RUDY/RUDY_long",   # long nets  → proxy for V demand
    "rudy_total":   "RUDY/RUDY",        # total RUDY
    "macro_region": "macro_region",     # binary blockage mask
}

# Fallback if short/long not present (happens for some designs)
FEATURE_SUBDIRS_FALLBACK = {
    "pin_density":  "RUDY/RUDY_pin",
    "cell_density": "cell_density",
    "rudy_h":       "RUDY/RUDY",
    "rudy_v":       "RUDY/RUDY",
    "rudy_total":   "RUDY/RUDY",
    "macro_region": "macro_region",
}

# Real CircuitNet Global Routing Utilization Congestion Targets
TARGET_SUBDIRS = {
    "congestion_h": "congestion/congestion_global_routing/utilization_based/congestion_GR_horizontal_util",
    "congestion_v": "congestion/congestion_global_routing/utilization_based/congestion_GR_vertical_util",
}

# Optional overflow targets
TARGET_OVERFLOW_SUBDIRS = {
    "overflow_h": "congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow",
    "overflow_v": "congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow",
}


class CircuitNetLoader:
    """
    Downloads & preprocesses CircuitNet N14 routability data into our
    standard 6-channel (features) + 2-channel (target) .npz format.
    """

    REPO_ID = "CircuitNet/CircuitNet"
    AVAILABLE_DESIGNS = [
        "nvdla-small",
        "nvdla-large",
        "zero-riscy",
        "RISCY",
        "RISCY-FPU",
        "Vortex-small",
        "Vortex-large",
        "openc910-1",
    ]

    def __init__(
        self,
        raw_dir: str = "./data/circuitnet_raw",
        processed_dir: str = "./data/processed",
        grid_h: int = 128,
        grid_w: int = 128,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────

    def download_design(self, design_name: str = "nvdla-small", hf_token: Optional[str] = None) -> Path:
        """Downloads a design's routability tar.gz from HuggingFace."""
        if not HF_AVAILABLE:
            raise ImportError("Install huggingface_hub: pip install huggingface_hub")

        hf_path = f"CircuitNet-N14/routability_features/{design_name}.tar.gz"
        print(f"[HF] Downloading {self.REPO_ID} → {hf_path} ...")

        local_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=hf_path,
            repo_type="dataset",
            local_dir=str(self.raw_dir),
            token=hf_token,
        )
        print(f"[✓] Downloaded → {local_path}")
        return Path(local_path)

    # ── Extract ───────────────────────────────────────────────────────────────

    def extract_archive(self, archive_path: Path) -> Path:
        """Extracts the tar.gz into raw_dir. Returns the extracted design folder."""
        design_name = archive_path.stem.replace(".tar", "")
        extract_to = self.raw_dir / design_name

        if extract_to.exists():
            print(f"[~] Already extracted: {extract_to}")
            return extract_to

        print(f"[*] Extracting {archive_path.name} ...")
        with tarfile.open(str(archive_path), "r:gz") as tar:
            tar.extractall(path=str(self.raw_dir))

        # Find the extracted folder
        for d in self.raw_dir.iterdir():
            if d.is_dir() and design_name.lower() in d.name.lower():
                return d

        return self.raw_dir / design_name  # best guess

    # ── Load one .npz feature map ─────────────────────────────────────────────

    @staticmethod
    def _load_npz(filepath: Path) -> Optional[np.ndarray]:
        """Loads a CircuitNet .npz file (key='data') as float32."""
        try:
            d = np.load(str(filepath))
            arr = d["data"].astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            return arr
        except Exception as e:
            print(f"  [!] Error loading {filepath}: {e}")
            return None

    @staticmethod
    def _resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """Bilinear resize via scipy.ndimage.zoom."""
        from scipy.ndimage import zoom
        zy = h / arr.shape[0]
        zx = w / arr.shape[1]
        return zoom(arr, (zy, zx), order=1).astype(np.float32)

    @staticmethod
    def _robust_norm(arr: np.ndarray) -> np.ndarray:
        """Normalize to [0,1]."""
        m = np.max(np.abs(arr))
        return (arr / (m + 1e-6)) if m > 0 else arr

    # ── Discover variants ─────────────────────────────────────────────────────

    def _get_variant_stems(self, design_root: Path) -> List[str]:
        """
        Finds all placement variant stems by listing the RUDY/RUDY folder.
        Each .npz there = one variant.
        """
        ref_folder = design_root / "RUDY" / "RUDY"
        if not ref_folder.exists():
            ref_folder = design_root / "cell_density"
        stems = sorted([p.stem for p in ref_folder.glob("*.npz")])
        return stems

    # ── Determine which feature subdirs to use ────────────────────────────────

    def _get_feature_map(self, design_root: Path) -> dict:
        """Use RUDY_short/long if available, otherwise fall back to plain RUDY."""
        if (design_root / "RUDY" / "RUDY_short").exists():
            return FEATURE_SUBDIRS
        return FEATURE_SUBDIRS_FALLBACK

    # ── Convert all variants to .npz ──────────────────────────────────────────

    def convert_all_variants(
        self,
        design_root: Path,
        design_name: str,
        max_samples: int = 96,
        overwrite: bool = False,
    ) -> int:
        """
        Converts each placement variant into a 6-channel feature + 2-channel target .npz.
        Returns the number of files successfully created.
        """
        stems = self._get_variant_stems(design_root)
        stems = stems[:max_samples]
        feature_map = self._get_feature_map(design_root)

        print(f"[*] {design_name}: {len(stems)} variants found")
        created = 0

        for i, stem in enumerate(stems):
            out_path = self.processed_dir / f"{design_name}_{i:04d}.npz"
            if out_path.exists() and not overwrite:
                created += 1
                continue

            # ── Build 6-channel feature ───────────────────────────────────────
            feature_channels = []
            ok = True
            for ch_name, subdir in feature_map.items():
                npz_file = design_root / subdir / f"{stem}.npz"
                arr = self._load_npz(npz_file) if npz_file.exists() else None

                if arr is None:
                    arr = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
                else:
                    # Resize to target grid
                    if arr.shape != (self.grid_h, self.grid_w):
                        arr = self._resize(arr, self.grid_h, self.grid_w)
                    if ch_name == "macro_region":
                        arr = (arr > 0.5).astype(np.float32)
                    else:
                        arr = self._robust_norm(arr)

                feature_channels.append(arr)

            # ── Build 2-channel target ────────────────────────────────────────
            target_channels = []
            for ch_name, subdir in TARGET_SUBDIRS.items():
                npz_file = design_root / subdir / f"{stem}.npz"
                arr = self._load_npz(npz_file) if npz_file.exists() else None

                if arr is None:
                    arr = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
                else:
                    if arr.shape != (self.grid_h, self.grid_w):
                        arr = self._resize(arr, self.grid_h, self.grid_w)
                    arr = self._robust_norm(arr)

                target_channels.append(arr)

            features = np.stack(feature_channels, axis=0).astype(np.float32)  # (6, H, W)
            target   = np.stack(target_channels,  axis=0).astype(np.float32)  # (2, H, W)

            np.savez_compressed(str(out_path), features=features, target=target)
            created += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(stems):
                print(f"  [{i+1:3d}/{len(stems)}] processed → {out_path.name}")

        print(f"[✓] {created} .npz files ready in {self.processed_dir}")
        return created

    # ── High-level API ────────────────────────────────────────────────────────

    def download_and_prepare(
        self,
        design_name: str = "nvdla-small",
        hf_token: Optional[str] = None,
        max_samples: int = 96,
        skip_download_if_exists: bool = True,
        overwrite: bool = False,
    ) -> int:
        """
        Full pipeline: Download → Extract → Convert to .npz
        Returns number of processed samples.
        """
        # 1. Locate or download archive
        archive_path = (
            self.raw_dir / "CircuitNet-N14" / "routability_features" / f"{design_name}.tar.gz"
        )
        if not archive_path.exists() or not skip_download_if_exists:
            archive_path = self.download_design(design_name, hf_token)

        # 2. Extract
        design_root = self.extract_archive(archive_path)

        # 3. Convert
        return self.convert_all_variants(design_root, design_name, max_samples, overwrite=overwrite)
