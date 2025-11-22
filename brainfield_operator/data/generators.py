# brainfield_operator/data/generators.py

from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np

from brainfield_operator.pde.geometry import create_cartesian_grid, create_layered_head_mask
from brainfield_operator.pde.electrodes import create_electrode_mask, random_electrode_pair
from brainfield_operator.pde.conductivity import create_conductivity_map
from brainfield_operator.pde.boundary_conditions import build_dirichlet_bc_from_electrodes
from brainfield_operator.pde.solver_fd import solve_poisson_fd
from brainfield_operator.pde.postprocess import compute_electric_field


@dataclass
class SimulationConfig:
    nx: int = 64
    ny: int = 64
    length: float = 0.2  # [m]
    brain_radius: float = 0.06
    skull_thickness: float = 0.007
    scalp_thickness: float = 0.008
    sigma_brain: float = 0.33
    sigma_skull: float = 0.015
    sigma_scalp: float = 0.43
    electrode_radius: float = 0.01
    anode_potential: float = 1.0
    cathode_potential: float = -1.0
    max_iter: int = 4000
    tol: float = 1e-5
    seed: int = 42


def simulate_single_sample(config: SimulationConfig, rng: np.random.Generator | None = None):
    """
    Generate one PDE simulation sample:
      - geometry & conductivity map
      - random tDCS-like electrode pair
      - solve Poisson PDE for potential V
      - compute electric field E

    Returns:
      dict with:
        sigma: (nx, ny)
        electrode_potential: (nx, ny) potential map on electrodes (0 elsewhere)
        V: (nx, ny) potential field
        Ex, Ey: (nx, ny) electric field components
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    # grid & head mask
    x, y, dx, dy = create_cartesian_grid(config.nx, config.ny, config.length)
    mask = create_layered_head_mask(
        x,
        y,
        brain_radius=config.brain_radius,
        skull_thickness=config.skull_thickness,
        scalp_thickness=config.scalp_thickness,
    )

    sigma = create_conductivity_map(
        mask,
        sigma_brain=config.sigma_brain,
        sigma_skull=config.sigma_skull,
        sigma_scalp=config.sigma_scalp,
    )

    # head outer radius (for electrode placement)
    head_radius = config.brain_radius + config.skull_thickness + config.scalp_thickness

    # random electrode pair
    anode_cfg, cathode_cfg = random_electrode_pair(
        x,
        y,
        head_radius=head_radius,
        electrode_radius=config.electrode_radius,
        anode_potential=config.anode_potential,
        cathode_potential=config.cathode_potential,
        rng=rng,
    )

    scalp_mask = mask == 3
    anode_mask = create_electrode_mask(x, y, scalp_mask, anode_cfg)
    cathode_mask = create_electrode_mask(x, y, scalp_mask, cathode_cfg)

    electrode_masks = [anode_mask, cathode_mask]
    potentials = [anode_cfg.potential, cathode_cfg.potential]

    bc_mask, bc_values = build_dirichlet_bc_from_electrodes(
        sigma.shape, electrode_masks, potentials
    )

    # solve PDE
    V = solve_poisson_fd(
        sigma,
        bc_mask,
        bc_values,
        dx,
        dy,
        max_iter=config.max_iter,
        tol=config.tol,
        verbose=False,
    )

    Ex, Ey = compute_electric_field(V, dx, dy)

    # build electrode potential map for input channel
    electrode_potential = np.zeros_like(V)
    electrode_potential[anode_mask] = anode_cfg.potential
    electrode_potential[cathode_mask] = cathode_cfg.potential

    sample = {
        "sigma": sigma.astype(np.float32),
        "electrode_potential": electrode_potential.astype(np.float32),
        "V": V.astype(np.float32),
        "Ex": Ex.astype(np.float32),
        "Ey": Ey.astype(np.float32),
    }
    return sample


def generate_dataset(
    n_samples: int,
    output_dir: str,
    config: SimulationConfig | None = None,
    prefix: str = "sample",
    start_index: int = 0,
):
    """
    Generate a dataset of PDE simulations and save each as .npz.

    Files will be saved as:
        {output_dir}/{prefix}_{index:05d}.npz

    Args:
        n_samples: number of simulations
        output_dir: directory to save
        config: SimulationConfig (default: SimulationConfig())
        prefix: file name prefix
        start_index: starting index for file numbering
    """
    if config is None:
        config = SimulationConfig()

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    for i in range(n_samples):
        # for reproducibility but variation: jump seed each iteration
        local_rng = np.random.default_rng(rng.integers(0, 1_000_000_000))
        sample = simulate_single_sample(config, rng=local_rng)
        idx = start_index + i
        path = os.path.join(output_dir, f"{prefix}_{idx:05d}.npz")
        np.savez_compressed(path, **sample)
        print(f"[generate_dataset] Saved {path}")
