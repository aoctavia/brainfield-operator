# brainfield_operator/pde/geometry.py

from __future__ import annotations
import numpy as np


def create_cartesian_grid(nx: int, ny: int, length: float):
    """
    Create a 2D Cartesian grid centered at (0, 0).

    Args:
        nx: number of grid points in x
        ny: number of grid points in y
        length: physical length of the domain (assumed square) in meters

    Returns:
        x: (nx, ny) array of x coordinates
        y: (nx, ny) array of y coordinates
        dx: grid spacing in x
        dy: grid spacing in y
    """
    x_lin = np.linspace(-length / 2.0, length / 2.0, nx)
    y_lin = np.linspace(-length / 2.0, length / 2.0, ny)
    dx = x_lin[1] - x_lin[0]
    dy = y_lin[1] - y_lin[0]
    x, y = np.meshgrid(x_lin, y_lin, indexing="ij")
    return x, y, dx, dy


def create_layered_head_mask(
    x: np.ndarray,
    y: np.ndarray,
    brain_radius: float = 0.06,
    skull_thickness: float = 0.007,
    scalp_thickness: float = 0.008,
) -> np.ndarray:
    """
    Create a simple 2D layered circular head model.

    Layers (integer labels):
        0: outside head
        1: brain
        2: skull
        3: scalp

    Args:
        x, y: coordinate grids (nx, ny)
        brain_radius: radius of brain region [m]
        skull_thickness: thickness of skull layer [m]
        scalp_thickness: thickness of scalp layer [m]

    Returns:
        mask: (nx, ny) integer array with layer labels
    """
    r = np.sqrt(x**2 + y**2)
    mask = np.zeros_like(r, dtype=np.int32)

    r1 = brain_radius
    r2 = brain_radius + skull_thickness
    r3 = brain_radius + skull_thickness + scalp_thickness

    mask[r <= r1] = 1  # brain
    mask[(r > r1) & (r <= r2)] = 2  # skull
    mask[(r > r2) & (r <= r3)] = 3  # scalp

    return mask
