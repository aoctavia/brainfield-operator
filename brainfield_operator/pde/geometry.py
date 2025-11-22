def create_cartesian_grid(nx: int, ny: int, length: float):
    """Return x, y coordinate grids and spacing dx, dy."""
    ...

def create_layered_head_mask(nx: int, ny: int,
                             brain_radius: float,
                             skull_thickness: float,
                             scalp_thickness: float):
    """Return integer mask: 0=outside, 1=brain, 2=skull, 3=scalp."""
    ...
