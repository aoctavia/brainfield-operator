import torch
from torch.utils.data import Dataset

class BrainFieldDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.files = ...  # list of file paths

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = ...  # load npz or pt
        x = ...       # stack sigma, electrode maps → [C_in, nx, ny]
        y = ...       # V field → [1, nx, ny]
        if self.transform:
            x, y = self.transform(x, y)
        return x.float(), y.float()
