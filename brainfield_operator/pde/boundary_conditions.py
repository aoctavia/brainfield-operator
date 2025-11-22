# brainfield_operator/pde/boundary_conditions.py

from __future__ import annotations
import numpy as np


def build_dirichlet_bc_from_electrodes(
    shape: tuple[int, int],
    electrode_masks: list[np.ndarray],
    potentials: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Dirichlet boundary conditions from electrode masks.

    Args:
        shape: (nx, ny)
        electrode_masks: list of boolean masks
        potentials: list of potentials (Same length as electrode_masks)

    Returns:
        bc_mask: bool array where potential is fixed
        bc_values: array with potential values where bc_mask True
    """
    assert len(electrode_masks) == len(potentials)
    nx, ny = shape
    bc_mask = np.zeros((nx, ny), dtype=bool)
    bc_values = np.zeros((nx, ny), dtype=np.float64)

    for mask, V in zip(electrode_masks, potentials):
        bc_mask |= mask
        bc_values[mask] = V

    return bc_mask, bc_values
