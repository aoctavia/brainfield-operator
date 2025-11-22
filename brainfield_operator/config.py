# brainfield_operator/config.py

from dataclasses import dataclass

@dataclass
class GridConfig:
    nx: int = 64
    ny: int = 64
    length: float = 0.2  # meter

@dataclass
class ConductivityConfig:
    sigma_brain: float = 0.33
    sigma_skull: float = 0.015
    sigma_scalp: float = 0.43

@dataclass
class PDEConfig:
    solver: str = "fd"   # finite-difference
    max_iter: int = 5000
    tol: float = 1e-6

@dataclass
class TrainingConfig:
    batch_size: int = 8
    lr: float = 1e-3
    num_epochs: int = 100
    model_type: str = "fno2d"
    device: str = "cuda"

@dataclass
class ExperimentConfig:
    grid: GridConfig = GridConfig()
    conductivity: ConductivityConfig = ConductivityConfig()
    pde: PDEConfig = PDEConfig()
    training: TrainingConfig = TrainingConfig()
