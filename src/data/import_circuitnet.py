"""
CircuitNet Data Import Script.
Run this after downloading the CircuitNet tar.gz to convert real EDA data
into our .npz training format.

Usage:
    # If tar.gz already downloaded:
    python src/data/import_circuitnet.py --design nvdla-small --skip_download

    # Download + convert in one shot:
    python src/data/import_circuitnet.py --design nvdla-small --hf_token YOUR_TOKEN
"""

import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, root_dir)

from src.data.circuitnet_loader import CircuitNetLoader


def main():
    parser = argparse.ArgumentParser(description="Import CircuitNet data from HuggingFace")
    parser.add_argument("--design", type=str, default="nvdla-small",
                        choices=CircuitNetLoader.AVAILABLE_DESIGNS,
                        help="Design name to download/process")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (optional, for higher rate limits)")
    parser.add_argument("--raw_dir", type=str, default="./data/circuitnet_raw",
                        help="Directory to store downloaded archives")
    parser.add_argument("--processed_dir", type=str, default="./data/circuitnet_processed",
                        help="Directory to save .npz files")
    parser.add_argument("--max_samples", type=int, default=96,
                        help="Maximum number of placement variants to convert")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download if archive already exists")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing processed .npz files")
    parser.add_argument("--grid_h", type=int, default=128)
    parser.add_argument("--grid_w", type=int, default=128)
    args = parser.parse_args()

    loader = CircuitNetLoader(
        raw_dir=os.path.join(root_dir, args.raw_dir),
        processed_dir=os.path.join(root_dir, args.processed_dir),
        grid_h=args.grid_h,
        grid_w=args.grid_w,
    )

    n = loader.download_and_prepare(
        design_name=args.design,
        hf_token=args.hf_token,
        max_samples=args.max_samples,
        skip_download_if_exists=args.skip_download,
        overwrite=args.overwrite,
    )

    print(f"\n✅ Import complete: {n} samples ready in {args.processed_dir}")
    print("   Now run: python src/pipeline/train.py")


if __name__ == "__main__":
    main()
