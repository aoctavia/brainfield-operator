# generate_dataset.py

"""
Generate a dataset of PDE simulations (tDCS-like brain fields)
using the brainfield_operator package.

Usage (example):
    python generate_dataset.py --n_samples 200 --output_dir data/brainfield
"""

import argparse
from brainfield_operator.data import SimulationConfig, generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PDE dataset for brain fields.")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of PDE simulations to generate.")
    parser.add_argument("--output_dir", type=str, default="data/brainfield",
                        help="Output directory for .npz files.")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--length", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="sample")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = SimulationConfig(
        nx=args.nx,
        ny=args.ny,
        length=args.length,
        seed=args.seed,
    )

    print(f"[generate_dataset] Generating {args.n_samples} samples to {args.output_dir}")
    generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        config=cfg,
        prefix=args.prefix,
        start_index=args.start_index,
    )
    print("[generate_dataset] Done.")


if __name__ == "__main__":
    main()
