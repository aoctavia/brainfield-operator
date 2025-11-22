# eval_operator.py

import torch
import os
import numpy as np

from brainfield_operator.data import BrainFieldDataset
from brainfield_operator.models import FNO2d, UNet2D
from brainfield_operator.utils import load_checkpoint
from brainfield_operator.visualization.plots import plot_comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained operator model.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="fno2d")
    parser.add_argument("--index", type=int, default=0, help="sample index to visualize")
    parser.add_argument("--save_path", type=str, default="figures/comparison_example.png")
    args = parser.parse_args()

    # Load dataset
    dataset = BrainFieldDataset(root_dir=args.data_dir)
    print(f"[eval] Loaded dataset: {len(dataset)} samples")

    # Load model
    if args.model_type.lower() == "fno2d":
        model = FNO2d(in_channels=2, out_channels=1, width=64, modes_x=16, modes_y=16)
    elif args.model_type.lower() == "unet2d":
        model = UNet2D(in_channels=2, out_channels=1)
    else:
        raise ValueError("Unknown model type")

    print(f"[eval] Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model)

    # Prepare data
    x, y = dataset[args.index]
    x = x.unsqueeze(0)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    x = x.to(device)

    # Predict
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0, 0]

    V_true = y.numpy()[0]

    # Plot comparison
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    plot_comparison(V_true, pred, save_path=args.save_path)
    print(f"[eval] Saved comparison plot to {args.save_path}")


if __name__ == "__main__":
    main()
