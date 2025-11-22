# brainfield_operator/pde/electrodes.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ElectrodeConfig:
    center_x: float
    center_y: float
    radius: float
    potential: float  # Volts


def create_electrode_mask(
    x: np.ndarray,
    y: np.ndarray,
    scalp_mask: np.ndarray,
    config: ElectrodeConfig,
) -> np.ndarray:
    """
    Create a boolean mask for an electrode patch, restricted to scalp.

    Args:
        x, y: coordinate grids (nx, ny)
        scalp_mask: boolean array True where scalp is present (mask == 3)
        config: ElectrodeConfig

    Returns:
        mask: boolean array for electrode region
    """
    dx = x - config.center_x
    dy = y - config.center_y
    r = np.sqrt(dx**2 + dy**2)
    mask = (r <= config.radius) & scalp_mask
    return mask


def random_electrode_pair(
    x: np.ndarray,
    y: np.ndarray,
    head_radius: float,
    electrode_radius: float = 0.01,
    anode_potential: float = 1.0,
    cathode_potential: float = -1.0,
    rng: np.random.Generator | None = None,
):
    """
    Sample two electrode positions on the scalp circle.

    Returns:
        anode_cfg, cathode_cfg: ElectrodeConfig
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample two distinct angles
    theta1 = rng.uniform(0, 2 * np.pi)
    theta2 = theta1 + np.pi + rng.uniform(-0.4, 0.4)  # roughly opposite

    # positions on circle
    cx1 = head_radius * np.cos(theta1)
    cy1 = head_radius * np.sin(theta1)
    cx2 = head_radius * np.cos(theta2)
    cy2 = head_radius * np.sin(theta2)

    anode_cfg = ElectrodeConfig(
        center_x=cx1, center_y=cy1, radius=electrode_radius, potential=anode_potential
    )
    cathode_cfg = ElectrodeConfig(
        center_x=cx2, center_y=cy2, radius=electrode_radius, potential=cathode_potential
    )

    return anode_cfg, cathode_cfg
