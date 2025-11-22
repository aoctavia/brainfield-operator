# brainfield_operator/training/train_operator.py

from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader
from brainfield_operator.training.metrics import mse_loss, l2_relative_error
from brainfield_operator.utils.logging import get_logger
from brainfield_operator.utils.seed import set_seed
from brainfield_operator.utils.io import save_checkpoint


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    mse_total = 0.0
    l2_total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_total += mse_loss(pred, y).item() * x.size(0)
            l2_total += l2_relative_error(pred, y).item() * x.size(0)
    n = len(loader.dataset)
    return mse_total / n, l2_total / n


def fit_operator_model(
    model,
    train_dataset,
    val_dataset,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_epochs: int = 50,
    device: str = "cuda",
    save_dir: str = "checkpoints",
    seed: int = 42,
):

    logger = get_logger("train")
    set_seed(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mse = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mse, val_l2 = validate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_mse={val_mse:.6f} | val_relL2={val_l2:.6f}"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            ckpt_path = os.path.join(save_dir, f"best_model_epoch{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch)

    logger.info(f"Training done. Best val MSE = {best_val_mse:.6f}")
