import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        ...

    def forward(self, x):
        ...

class FNO2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,
                 width=64, modes_x=12, modes_y=12):
        super().__init__()
        ...
    
    def forward(self, x):
        """
        x: [B, C_in, nx, ny]
        return: [B, 1, nx, ny] predicted potential V_hat
        """
        ...
