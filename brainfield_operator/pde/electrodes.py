from dataclasses import dataclass

@dataclass
class ElectrodeConfig:
    center_x: float
    center_y: float
    radius: float
    potential: float  # Volt

def create_electrode_mask(x, y, config: ElectrodeConfig):
    """Return boolean mask array for electrode area."""
    ...

def random_electrode_pair(x, y):
    """Sample random anode/cathode positions on scalp region."""
    ...
