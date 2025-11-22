# train_operator.py

"""
Train a neural operator surrogate (FNO2D / UNet2D) to approximate
the PDE solution of brain electric potentials.

Usage examples:

    # Simple default training using FNO and data from data/brainfield
    python train_operator.py --data_dir data/brainfield --model_type fno2d

    # Use YAML experiment config
    python train_operator.py --data_dir data/brainfield --config experiments/exp_fno_training.yaml
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import random_split

from brainfield_operator.data import BrainFieldDataset
from brainfield_operator.models import FNO2d, UNet2D
from brainfield_operator.training import fit_operator_model
from brainfield_operator.utils import get_logger, set_seed


def load_yaml_config(path: str) -> dict:
    """
    Load YAML config file as python dict.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


from typing import Optional, Dict

def build_model_from_config(model_type: str, cfg: Optional[Dict] = None):

    """
    Instantiate model based on model_type and optional extra config.
    """
    if model_type.lower() == "fno2d":
        fno_cfg = (cfg or {}).get("fno", {})
        model = FNO2d(
            in_channels=fno_cfg.get("in_channels", 2),
            out_channels=fno_cfg.get("out_channels", 1),
            width=fno_cfg.get("width", 64),
            modes_x=fno_cfg.get("modes_x", 16),
            modes_y=fno_cfg.get("modes_y", 16),
            n_layers=fno_cfg.get("n_layers", 4),
        )
    elif model_type.lower() == "unet2d":
        unet_cfg = (cfg or {}).get("unet", {})
        model = UNet2D(
            in_channels=unet_cfg.get("in_channels", 2),
            out_channels=unet_cfg.get("out_channels", 1),
            base_channels=unet_cfg.get("base_channels", 32),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train neural operator surrogate.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .npz samples.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML experiment config (optional).")
    parser.add_argument("--model_type", type=str, default="fno2d",
                        help="Model type: fno2d or unet2d.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data used for validation.")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("main")

    # Load YAML config if provided
    yaml_cfg = None
    if args.config is not None:
        logger.info(f"Loading config from {args.config}")
        yaml_cfg = load_yaml_config(args.config)

    # Merge basic training hyperparams (CLI has priority)
    train_cfg = (yaml_cfg or {}).get("training", {})
    batch_size = args.batch_size if "batch_size" not in train_cfg else train_cfg["batch_size"]
    lr = args.lr if "lr" not in train_cfg else float(train_cfg["lr"])
    num_epochs = args.num_epochs if "num_epochs" not in train_cfg else int(train_cfg["num_epochs"])
    device = args.device if "device" not in train_cfg else train_cfg["device"]
    seed = args.seed if "seed" not in train_cfg else int(train_cfg["seed"])

    set_seed(seed)

    # Dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    full_dataset = BrainFieldDataset(root_dir=args.data_dir)
    n_total = len(full_dataset)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    logger.info(f"Dataset split: {n_train} train, {n_val} val")

    # Model
    model_type = args.model_type if args.config is None else (yaml_cfg.get("training", {}).get("model_type", args.model_type))
    logger.info(f"Building model: {model_type}")
    model = build_model_from_config(model_type, yaml_cfg)

    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(
        f"Starting training | model={model_type}, batch_size={batch_size}, "
        f"lr={lr}, epochs={num_epochs}, device={device}"
    )

    fit_operator_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        device=device,
        save_dir=args.save_dir,
        seed=seed,
    )


if __name__ == "__main__":
    main()
