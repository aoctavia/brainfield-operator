# animate_comparison.py

import numpy as np
import torch
import os

from brainfield_operator.data import BrainFieldDataset
from brainfield_operator.utils import load_checkpoint
from brainfield_operator.models import FNO2d, UNet2D
from brainfield_operator.visualization.animations import animate_comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Animate Ground Truth vs Predicted Comparison GIF.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .npz PDE samples.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--model_type", type=str, default="fno2d",
                        help="fno2d or unet2d")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="figures/comparison.gif")
    args = parser.parse_args()

    # Load dataset
    dataset = BrainFieldDataset(root_dir=args.data_dir)
    print(f"[comparison] Loaded dataset: {len(dataset)} samples")

    # Load model
    if args.model_type.lower() == "fno2d":
        model = FNO2d(in_channels=2, out_channels=1, width=64, modes_x=16, modes_y=16)
    elif args.model_type.lower() == "unet2d":
        model = UNet2D(in_channels=2, out_channels=1)
    else:
        raise ValueError("Unknown model type")

    print(f"[comparison] Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model)

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Prepare frames
    V_true_list = []
    V_pred_list = []

    print("[comparison] Generating prediction frames...")

    for idx in range(args.num_frames):
        x, y = dataset[idx]         # True data
        V_true = y.numpy()[0]       # ground truth

        # Predict
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0, 0]

        V_true_list.append(V_true)
        V_pred_list.append(pred)

    # Create animation
    print("[comparison] Creating animation...")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    animate_comparison(
        V_true_list,
        V_pred_list,
        interval=250,
        save_path=args.save_path
    )

    print(f"[comparison] Saved animation to {args.save_path}")


if __name__ == "__main__":
    main()
