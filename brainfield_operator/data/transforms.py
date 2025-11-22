# brainfield_operator/data/transforms.py

from __future__ import annotations
import torch


class NormalizeInputs:
    """
    Normalize inputs channel-wise: (x - mean) / std.

    mean and std are computed per-channel over the batch on-the-fly.
    For small projects this is often enough; for serious work you can
    pre-compute dataset statistics.
    """

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        # x: [C, H, W]
        c, _, _ = x.shape
        for i in range(c):
            ch = x[i]
            mean = ch.mean()
            std = ch.std()
            if std > 1e-6:
                x[i] = (ch - mean) / std
            else:
                x[i] = ch - mean
        return x, y


class AddNoiseToTarget:
    """
    Optional: add small Gaussian noise to V during training for robustness.
    """

    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.sigma > 0:
            noise = torch.randn_like(y) * self.sigma
            y = y + noise
        return x, y
