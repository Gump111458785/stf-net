import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierTemporalMixing(nn.Module):
    """
    Learned complex spectral filtering along the temporal frequency domain.
    Applies rfft along time dimension, parameterizes frequency modes with complex weights,
    and reconstructs temporal patterns via irfft.
    """
    def __init__(self, seq_len, modes=32):
        super().__init__()
        self.seq_len = seq_len
        self.modes = min(modes, seq_len // 2 + 1)
        # Complex weights for the top frequency modes
        self.weights_real = nn.Parameter(torch.randn(self.modes) * 0.02)
        self.weights_imag = nn.Parameter(torch.randn(self.modes) * 0.02)

    def forward(self, x):
        # x: [Batch, Time, Channels]
        B, T, C = x.shape
        # Permute to [B, C, T] for 1D FFT along time
        x_perm = x.permute(0, 2, 1)
        x_ft = torch.fft.rfft(x_perm, dim=-1)
        
        # Multiply dominant frequency modes
        modes = self.modes
        out_ft = torch.zeros_like(x_ft)
        
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        w = torch.complex(self.weights_real, self.weights_imag)
        out_ft[:, :, :modes] = x_ft[:, :, :modes] * w.unsqueeze(0).unsqueeze(0)
        
        # Inverse real FFT
        x_out = torch.fft.irfft(out_ft, n=T, dim=-1)
        return x_out.permute(0, 2, 1) + x


class SpatialChannelInteraction(nn.Module):
    """
    Adaptive cross-variate channel interaction module.
    Models inter-dependencies among different multivariate sensor series.
    """
    def __init__(self, num_channels, hidden_dim=64):
        super().__init__()
        self.channel_proj1 = nn.Linear(num_channels, hidden_dim)
        self.channel_proj2 = nn.Linear(hidden_dim, num_channels)
        self.norm = nn.LayerNorm(num_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [Batch, Time, Channels]
        residual = x
        x_norm = self.norm(x)
        out = self.channel_proj2(self.act(self.channel_proj1(x_norm)))
        return residual + out


class STFNet(nn.Module):
    """
    Spatio-Temporal Fourier Neural Network for Multivariate Long-Sequence Forecasting.
    Combines:
    1. Temporal Fourier frequency domain representation
    2. Adaptive cross-variate spatial interaction
    3. Direct horizon projection head
    """
    def __init__(self, seq_len=96, pred_len=96, num_channels=7, modes=32, hidden_dim=64, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_channels = num_channels

        self.fourier_blocks = nn.ModuleList([
            FourierTemporalMixing(seq_len=seq_len, modes=modes)
            for _ in range(num_layers)
        ])
        
        self.spatial_blocks = nn.ModuleList([
            SpatialChannelInteraction(num_channels=num_channels, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        self.temporal_proj = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [Batch, seq_len, num_channels]
        for fourier_layer, spatial_layer in zip(self.fourier_blocks, self.spatial_blocks):
            x = fourier_layer(x)
            x = spatial_layer(x)
            x = self.dropout(x)

        # Forecasting projection across temporal dimension
        # [B, T, C] -> [B, C, T] -> [B, C, pred_len] -> [B, pred_len, C]
        x_perm = x.permute(0, 2, 1)
        pred = self.temporal_proj(x_perm).permute(0, 2, 1)
        return pred
