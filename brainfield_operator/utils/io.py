# brainfield_operator/utils/io.py

import numpy as np
import torch
import os


def save_npz(path: str, **arrays):
    """
    Save multiple arrays as .npz
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[io] Saved npz file to {path}")


def load_npz(path: str):
    """
    Load npz file and return dict
    """
    print(f"[io] Loading npz file from {path}")
    return dict(np.load(path))


def save_checkpoint(path: str, model, optimizer=None, epoch: int = None):
    """
    Save PyTorch model checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model_state": model.state_dict()}
    if optimizer:
        ckpt["optim_state"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    torch.save(ckpt, path)
    print(f"[io] Checkpoint saved to {path}")


def load_checkpoint(path: str, model, optimizer=None):
    """
    Load PyTorch checkpoint.
    """
    print(f"[io] Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt.get("epoch", None)
