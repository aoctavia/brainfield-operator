# brainfield_operator/visualization/animations.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os


def animate_potential_sequence(V_list, interval=200, save_path=None):
    """
    Create an animation of potential fields (e.g., multiple samples).
    
    Args:
        V_list: list of 2D numpy arrays [H, W]
        interval: delay between frames in ms
        save_path: optional path to save (mp4/gif)
    Returns:
        anim: matplotlib.animation.FuncAnimation object
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(V_list[0], cmap="coolwarm", origin="lower")
    plt.colorbar(img, ax=ax, fraction=0.046)

    def update(i):
        img.set_array(V_list[i])
        ax.set_title(f"Frame {i}")
        return [img]

    anim = animation.FuncAnimation(
        fig, update, frames=len(V_list), interval=interval, blit=True
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1]
        if ext == ".gif":
            anim.save(save_path, writer="imagemagick")
        else:
            anim.save(save_path, writer="ffmpeg")
        print(f"[animation] saved to {save_path}")

    plt.close()
    return anim


def animate_comparison(V_true_list, V_pred_list, interval=200, save_path=None):
    """
    Animate comparison between true vs predicted potential fields.

    Args:
        V_true_list: list of ground-truth V arrays
        V_pred_list: list of predicted V arrays
    """
    assert len(V_true_list) == len(V_pred_list)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im_true = axes[0].imshow(V_true_list[0], cmap="coolwarm", origin="lower")
    axes[0].set_title("True")

    im_pred = axes[1].imshow(V_pred_list[0], cmap="coolwarm", origin="lower")
    axes[1].set_title("Predicted")

    im_err = axes[2].imshow(np.abs(V_true_list[0] - V_pred_list[0]),
                            cmap="inferno", origin="lower")
    axes[2].set_title("|Error|")

    plt.tight_layout()

    def update(i):
        im_true.set_array(V_true_list[i])
        im_pred.set_array(V_pred_list[i])
        im_err.set_array(np.abs(V_true_list[i] - V_pred_list[i]))
        return [im_true, im_pred, im_err]

    anim = animation.FuncAnimation(
        fig, update, frames=len(V_true_list), interval=interval, blit=True
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1]
        if ext == ".gif":
            anim.save(save_path, writer="imagemagick")
        else:
            anim.save(save_path, writer="ffmpeg")
        print(f"[animation] saved to {save_path}")

    plt.close()
    return anim


def animate_quiver(Ex_list, Ey_list, step=4, interval=200, save_path=None):
    """
    Animate electric field quiver arrows.

    Args:
        Ex_list, Ey_list: list of 2D arrays (electric field components)
    """
    assert len(Ex_list) == len(Ey_list)

    H, W = Ex_list[0].shape
    x = np.arange(0, W, step)
    y = np.arange(0, H, step)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    Q = ax.quiver(
        X,
        Y,
        Ex_list[0][::step, ::step],
        Ey_list[0][::step, ::step],
        color="black",
    )
    ax.set_title("Electric Field Animation")
    ax.invert_yaxis()

    def update(i):
        Q.set_UVC(
            Ex_list[i][::step, ::step],
            Ey_list[i][::step, ::step]
        )
        ax.set_title(f"Frame {i}")
        return Q,

    anim = animation.FuncAnimation(
        fig, update, frames=len(Ex_list), interval=interval, blit=False
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1]
        if ext == ".gif":
            anim.save(save_path, writer="imagemagick")
        else:
            anim.save(save_path, writer="ffmpeg")
        print(f"[animation] saved to {save_path}")

    plt.close()
    return anim
