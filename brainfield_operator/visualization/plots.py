# brainfield_operator/visualization/plots.py

import numpy as np
import matplotlib.pyplot as plt


def plot_potential_map(V, title=None, save_path=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(V, cmap="coolwarm", origin="lower")
    plt.colorbar(label="Potential (V)")
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_field_quiver(Ex, Ey, step=3, save_path=None):
    H, W = Ex.shape
    x = np.arange(0, W, step)
    y = np.arange(0, H, step)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, Ex[Y, X], Ey[Y, X], color="black", alpha=0.6)
    plt.title("Electric Field (Arrows)")
    plt.gca().invert_yaxis()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(V_true, V_pred, save_path=None):
    """
    Create visual comparison of PDE solution vs operator prediction.
    """
    error = np.abs(V_true - V_pred)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(V_true, cmap='coolwarm', origin='lower')
    axs[0].set_title("Ground Truth V")

    axs[1].imshow(V_pred, cmap='coolwarm', origin='lower')
    axs[1].set_title("Predicted V̂")

    axs[2].imshow(error, cmap='inferno', origin='lower')
    axs[2].set_title("|Error|")

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
