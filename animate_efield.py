# animate_efield.py

import numpy as np
import torch
from brainfield_operator.data import BrainFieldDataset
from brainfield_operator.models import FNO2d
from brainfield_operator.utils import load_checkpoint
from brainfield_operator.visualization.animations import animate_quiver


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Animate Electric Field (Ex, Ey) GIF.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .npz samples.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--save_path", type=str, default="figures/efield.gif")
    parser.add_argument("--num_frames", type=int, default=10)
    args = parser.parse_args()

    print("[anim] Loading dataset...")
    dataset = BrainFieldDataset(root_dir=args.data_dir)

    print("[anim] Loading model...")
    model = FNO2d(in_channels=2, out_channels=1, width=64, modes_x=16, modes_y=16)
    load_checkpoint(args.checkpoint, model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    Ex_list = []
    Ey_list = []

    print("[anim] Generating electric field frames...")

    for idx in range(args.num_frames):
        # load True PDE data (for this animation we don't need model output)
        sample_path = dataset.files[idx]
        data = np.load(sample_path)
        Ex = data["Ex"]
        Ey = data["Ey"]

        Ex_list.append(Ex)
        Ey_list.append(Ey)

    print("[anim] Creating GIF...")
    animate_quiver(
        Ex_list,
        Ey_list,
        step=3,
        save_path=args.save_path
    )

    print(f"[anim] Electric Field GIF saved to {args.save_path}")


if __name__ == "__main__":
    main()
