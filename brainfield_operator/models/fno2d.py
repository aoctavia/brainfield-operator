# brainfield_operator/models/fno2d.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.fft


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer; taken from the basic FNO implementation pattern.

    It does:
        - rFFT2
        - linear transform on a limited number of modes
        - irFFT2
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Complex weights for the Fourier modes
        self.scale = 1 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y)
        )
        self.weight_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y)
        )

    def compl_mul2d(self, input, weight_real, weight_imag):
        # input: [B, in_c, X, Y], weight: [in_c, out_c, X, Y]
        # output: [B, out_c, X, Y]
        # (a+bi)(c+di) = (ac - bd) + (ad+bc)i
        in_c = input.shape[1]
        real = input.real.unsqueeze(2)  # [B, in_c, 1, X, Y]
        imag = input.imag.unsqueeze(2)

        wr = weight_real.unsqueeze(0)  # [1, in_c, out_c, X, Y]
        wi = weight_imag.unsqueeze(0)

        out_real = (real * wr - imag * wi).sum(dim=1)  # [B, out_c, X, Y]
        out_imag = (real * wi + imag * wr).sum(dim=1)
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]
        """
        B, C, H, W = x.shape

        # Fourier transform
        x_ft = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]

        # Truncate to low modes
        out_ft = torch.zeros(
            B,
            self.out_channels,
            H,
            W // 2 + 1,
            dtype=torch.complex64,
            device=x.device,
        )

        mx = min(self.modes_x, H)
        my = min(self.modes_y, W // 2 + 1)

        x_ft_crop = x_ft[:, :, :mx, :my]
        w_real = self.weight_real[:, :, :mx, :my]
        w_imag = self.weight_imag[:, :, :mx, :my]

        out_ft[:, :, :mx, :my] = self.compl_mul2d(x_ft_crop, w_real, w_imag)

        # Back to physical space
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")  # [B, out_c, H, W]
        return x_out
        

class FNO2d(nn.Module):
    """
    Basic 2D Fourier Neural Operator model.

    Assumes input x: [B, C_in, H, W]
            output y: [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        width: int = 64,
        modes_x: int = 16,
        modes_y: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.n_layers = n_layers

        # Lift input to higher dimension
        self.input_layer = nn.Conv2d(in_channels, width, kernel_size=1)

        # Spectral layers
        self.spectral_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        for _ in range(n_layers):
            self.spectral_convs.append(SpectralConv2d(width, width, modes_x, modes_y))
            self.pointwise_convs.append(
                nn.Conv2d(width, width, kernel_size=1)
            )

        self.activation = nn.GELU()

        # Projection to output
        self.output_layer1 = nn.Conv2d(width, width // 2, kernel_size=1)
        self.output_layer2 = nn.Conv2d(width // 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_channels, H, W]
        """
        x = self.input_layer(x)  # [B, width, H, W]

        for spec_conv, pw_conv in zip(self.spectral_convs, self.pointwise_convs):
            x1 = spec_conv(x)
            x2 = pw_conv(x)
            x = self.activation(x1 + x2)

        x = self.activation(self.output_layer1(x))
        x = self.output_layer2(x)
        return x
