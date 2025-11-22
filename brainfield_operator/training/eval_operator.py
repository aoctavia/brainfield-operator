# brainfield_operator/training/eval_operator.py

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from brainfield_operator.training.metrics import mse_loss, l2_relative_error
from brainfield_operator.visualization.plots import plot_comparison
import os


def evaluate_on_test_set(model, test_dataset, device="cuda", save_dir="eval_results"):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    mse_list = []
    l2_list = []

    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)

        mse_val = mse_loss(pred, y).item()
        l2_val = l2_relative_error(pred, y).item()
        mse_list.append(mse_val)
        l2_list.append(l2_val)

        # Save comparison plot for first few samples
        if i < 5:
            plot_path = os.path.join(save_dir, f"compare_{i:03d}.png")
            plot_comparison(
                y.cpu().numpy()[0, 0],
                pred.cpu().numpy()[0, 0],
                save_path=plot_path,
            )

    return {
        "mean_mse": sum(mse_list) / len(mse_list),
        "mean_l2": sum(l2_list) / len(l2_list),
    }
