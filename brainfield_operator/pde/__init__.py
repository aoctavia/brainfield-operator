# brainfield_operator/pde/__init__.py

from .geometry import create_cartesian_grid, create_layered_head_mask
from .electrodes import ElectrodeConfig, create_electrode_mask, random_electrode_pair
from .conductivity import create_conductivity_map
from .boundary_conditions import build_dirichlet_bc_from_electrodes
from .solver_fd import solve_poisson_fd
from .postprocess import compute_electric_field, normalize_potential
