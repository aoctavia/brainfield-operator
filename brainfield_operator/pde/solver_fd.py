import numpy as np

def build_poisson_matrix(sigma, dx, dy, bc_mask):
    """
    Build sparse matrix A for A * V = b with heterogeneous conductivity sigma.
    """
    ...

def solve_poisson_fd(sigma, bc_mask, bc_values, dx, dy, max_iter=5000, tol=1e-6):
    """
    Solve ∇·(σ∇V)=0 on a 2D grid with Dirichlet boundary conditions.

    Returns:
      V: potential field [nx, ny]
    """
    ...
