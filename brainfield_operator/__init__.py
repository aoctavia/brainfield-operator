# brainfield_operator/__init__.py

"""
brainfield_operator
===================

A hybrid physics + machine-learning framework for modeling electric
potentials and fields inside the brain under external stimulation
(tDCS/tACS), including:

- PDE modeling (Poisson equation)
- Layered head geometry (brain, skull, scalp)
- Electrode configuration and boundary conditions
- Dataset generation for ML surrogate models
- FNO2D and UNet2D neural operator architectures
- Training, evaluation, visualization utilities
"""

# PDE components
from .pde import (
    create_cartesian_grid,
    create_layered_head_mask,
    ElectrodeConfig,
    create_electrode_mask,
    random_electrode_pair,
    create_conductivity_map,
    build_dirichlet_bc_from_electrodes,
    solve_poisson_fd,
    compute_electric_field,
    normalize_potential,
)

# Data generation
from .data import (
    SimulationConfig,
    simulate_single_sample,
    generate_dataset,
    BrainFieldDataset,
)

# Models
from .models import FNO2d, UNet2D

# Training
from .training import (
    mse_loss,
    l2_relative_error,
    fit_operator_model,
    evaluate_on_test_set,
)

# Utilities
from .utils import (
    set_seed,
    save_npz,
    load_npz,
    save_checkpoint,
    load_checkpoint,
    get_logger,
)
