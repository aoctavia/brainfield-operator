# brainfield_operator/config.py

from dataclasses import dataclass


# -----------------------------
# Grid & Geometry Configuration
# -----------------------------
@dataclass
class GridConfig:
    nx: int = 64
    ny: int = 64
    length: float = 0.20  # meters (side length of square domain)


@dataclass
class GeometryConfig:
    brain_radius: float = 0.06
    skull_thickness: float = 0.007
    scalp_thickness: float = 0.008


# -----------------------------
# Conductivity
# -----------------------------
@dataclass
class ConductivityConfig:
    sigma_brain: float = 0.33
    sigma_skull: float = 0.015
    sigma_scalp: float = 0.43


# -----------------------------
# PDE Solver Settings
# -----------------------------
@dataclass
class PDEConfig:
    max_iter: int = 4000
    tol: float = 1e-5
    solver: str = "gauss-seidel"  # placeholder, extensible


# -----------------------------
# Electrode Settings
# -----------------------------
@dataclass
class ElectrodeSystemConfig:
    electrode_radius: float = 0.01
    anode_potential: float = 1.0
    cathode_potential: float = -1.0


# -----------------------------
# ML Training Config
# -----------------------------
@dataclass
class TrainingConfig:
    batch_size: int = 8
    lr: float = 1e-3
    num_epochs: int = 50
    device: str = "cuda"
    model_type: str = "fno2d"
    seed: int = 42


# -----------------------------
# Main Experiment Configuration
# -----------------------------
@dataclass
class ExperimentConfig:
    grid: GridConfig = GridConfig()
    geometry: GeometryConfig = GeometryConfig()
    conductivity: ConductivityConfig = ConductivityConfig()
    pde: PDEConfig = PDEConfig()
    electrode: ElectrodeSystemConfig = ElectrodeSystemConfig()
    training: TrainingConfig = TrainingConfig()
