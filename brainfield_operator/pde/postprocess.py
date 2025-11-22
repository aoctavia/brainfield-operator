# brainfield_operator/pde/postprocess.py

from __future__ import annotations
import numpy as np


def compute_electric_field(V: np.ndarray, dx: float, dy: float):
    """
    Compute electric field components E = -∇V.

    Args:
        V: potential field (nx, ny)
        dx, dy: grid spacing

    Returns:
        Ex, Ey: electric field components (nx, ny)
    """
    dVdx = np.zeros_like(V)
    dVdy = np.zeros_like(V)

    # central differences interior, one-sided at boundaries
    dVdx[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2 * dx)
    dVdx[0, :] = (V[1, :] - V[0, :]) / dx
    dVdx[-1, :] = (V[-1, :] - V[-2, :]) / dx

    dVdy[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * dy)
    dVdy[:, 0] = (V[:, 1] - V[:, 0]) / dy
    dVdy[:, -1] = (V[:, -1] - V[:, -2]) / dy

    Ex = -dVdx
    Ey = -dVdy
    return Ex, Ey


def normalize_potential(V: np.ndarray):
    """
    Normalize potential for training/visualization:
    subtract mean and divide by std (if non-zero).
    """
    mean = V.mean()
    std = V.std()
    if std < 1e-8:
        return V - mean
    return (V - mean) / std
