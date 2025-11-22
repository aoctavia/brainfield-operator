import torch
from torch.utils.data import DataLoader
from brainfield_operator.models.fno2d import FNO2d
from brainfield_operator.data.dataset import BrainFieldDataset
from brainfield_operator.training.metrics import mse_loss

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = mse_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    ...
    return avg_mse, avg_rel_l2

def fit_operator_model(config, data_dir, save_path):
    """
    High-level training function:
      - load dataset & dataloader
      - instantiate model + optimizer
      - run epochs
      - save best model
    """
    ...
