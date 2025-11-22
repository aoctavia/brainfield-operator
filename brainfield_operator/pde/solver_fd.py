# brainfield_operator/pde/solver_fd.py

from __future__ import annotations
import numpy as np


def solve_poisson_fd(
    sigma: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    dx: float,
    dy: float,
    max_iter: int = 5000,
    tol: float = 1e-5,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve ∇·(σ ∇V) = 0 on a 2D grid using Gauss-Seidel iteration.

    Args:
        sigma: (nx, ny) conductivity map
        bc_mask: bool array where potential is fixed
        bc_values: array with fixed potential values
        dx, dy: grid spacing
        max_iter: maximum number of iterations
        tol: stopping tolerance on max update
        verbose: whether to print convergence info

    Returns:
        V: potential field (nx, ny)
    """
    nx, ny = sigma.shape
    V = np.zeros_like(sigma, dtype=np.float64)

    # apply initial BC
    V[bc_mask] = bc_values[bc_mask]

    dx2 = dx * dx
    dy2 = dy * dy

    for it in range(max_iter):
        max_update = 0.0

        # Gauss-Seidel sweep
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if bc_mask[i, j]:
                    continue

                sig_ij = sigma[i, j]
                if sig_ij == 0.0:
                    # outside head region
                    continue

                # conductivities to neighbors (average at interfaces)
                sxp = 0.5 * (sigma[i, j] + sigma[i + 1, j])
                sxm = 0.5 * (sigma[i, j] + sigma[i - 1, j])
                syp = 0.5 * (sigma[i, j] + sigma[i, j + 1])
                sym = 0.5 * (sigma[i, j] + sigma[i, j - 1])

                denom = sxp / dx2 + sxm / dx2 + syp / dy2 + sym / dy2
                if denom == 0.0:
                    continue

                num = (
                    sxp * V[i + 1, j] / dx2
                    + sxm * V[i - 1, j] / dx2
                    + syp * V[i, j + 1] / dy2
                    + sym * V[i, j - 1] / dy2
                )

                new_V = num / denom
                update = abs(new_V - V[i, j])
                if update > max_update:
                    max_update = update
                V[i, j] = new_V

        if verbose and (it % 100 == 0 or it == max_iter - 1):
            print(f"[Poisson] iter={it} max_update={max_update:.3e}")

        if max_update < tol:
            if verbose:
                print(f"[Poisson] converged in {it} iterations, max_update={max_update:.3e}")
            break

    # enforce BC again
    V[bc_mask] = bc_values[bc_mask]
    return V
