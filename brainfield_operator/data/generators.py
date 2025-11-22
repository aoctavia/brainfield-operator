from brainfield_operator.pde.geometry import create_cartesian_grid, create_layered_head_mask
from brainfield_operator.pde.electrodes import random_electrode_pair, create_electrode_mask
from brainfield_operator.pde.conductivity import create_conductivity_map
from brainfield_operator.pde.boundary_conditions import build_dirichlet_bc_from_electrodes
from brainfield_operator.pde.solver_fd import solve_poisson_fd
from brainfield_operator.pde.postprocess import compute_electric_field

def simulate_single_sample(config) -> dict:
    """
    Generate one (input, output) pair:
      input: {
        'sigma': conductivity map,
        'electrode_mask': mask anode/cathode or potential map
      }
      output: {
        'V': potential field,
        'E': electric field components (optional)
      }
    """
    ...

def generate_dataset(n_samples: int, output_dir: str, config) -> None:
    """
    Loop: simulate_single_sample n times and save as .npz or .pt files.
    """
    ...
