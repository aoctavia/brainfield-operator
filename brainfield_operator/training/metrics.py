import torch

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def l2_relative_error(pred, target):
    num = torch.norm(pred - target)
    den = torch.norm(target) + 1e-8
    return num / den
