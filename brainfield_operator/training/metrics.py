# brainfield_operator/training/metrics.py

import torch


def mse_loss(pred, target):
    """
    Mean squared error loss
    """
    return torch.mean((pred - target) ** 2)


def l2_relative_error(pred, target):
    """
    ||pred - target|| / ||target||
    """
    num = torch.norm(pred - target)
    den = torch.norm(target) + 1e-8
    return num / den
