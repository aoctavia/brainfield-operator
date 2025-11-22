# brainfield_operator/pde/conductivity.py

from __future__ import annotations
import numpy as np


def create_conductivity_map(
    mask: np.ndarray,
    sigma_brain: float = 0.33,
    sigma_skull: float = 0.015,
    sigma_scalp: float = 0.43,
) -> np.ndarray:
    """
    Create a piecewise-constant conductivity map based on layer mask.

    Args:
        mask: integer mask (0 outside, 1 brain, 2 skull, 3 scalp)
        sigma_brain, sigma_skull, sigma_scalp: conductivities [S/m]

    Returns:
        sigma: float array (nx, ny)
    """
    sigma = np.zeros_like(mask, dtype=np.float64)
    sigma[mask == 1] = sigma_brain
    sigma[mask == 2] = sigma_skull
    sigma[mask == 3] = sigma_scalp
    return sigma
