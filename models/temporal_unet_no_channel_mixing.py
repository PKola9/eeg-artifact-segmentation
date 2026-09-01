"""
Channel-independent temporal U-Net baseline for EEG artifact segmentation.

Purpose:
    This is a simple U-Net baseline that does NOT explicitly learn relationships
    between EEG channels.

Input:
    batch x electrodes x time

Output:
    batch x electrodes x time logits

How it avoids channel mixing:
    The model reshapes the input from:
        batch x electrodes x time
    into:
        (batch * electrodes) x 1 x time

    Then the same 1D temporal U-Net is applied independently to each electrode.
    This means each channel is segmented from its own waveform only.

Why useful:
    It gives a fair baseline for testing whether Split U-Net channel/electrode
    modeling helps compared with temporal-only processing.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """Two 1D temporal convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalDown(nn.Module):
    """Downsample temporal resolution by a factor of 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = TemporalConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.block(x)
        pooled = self.pool(features)
        return features, pooled


class TemporalUp(nn.Module):
    """Upsample temporal resolution and concatenate skip features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = TemporalConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class TemporalUNetNoChannelMixing(nn.Module):
    """
    Temporal-only U-Net applied independently to every EEG electrode.

    This is the "normal/simple U-Net" comparison model for the thesis:
        - it learns temporal artifact patterns;
        - it does not combine information across electrodes.
    """

    def __init__(self, base_features: int = 8):
        super().__init__()
        self.down1 = TemporalDown(1, base_features)
        self.down2 = TemporalDown(base_features, base_features * 2)
        self.bottleneck = TemporalConvBlock(base_features * 2, base_features * 4)
        self.up2 = TemporalUp(base_features * 4, base_features * 2, base_features * 2)
        self.up1 = TemporalUp(base_features * 2, base_features, base_features)
        self.out = nn.Conv1d(base_features, 1, kernel_size=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: batch x electrodes x time
        batch_size, electrodes, time_samples = eeg.shape

        # Treat every electrode as an independent temporal signal.
        x = eeg.reshape(batch_size * electrodes, 1, time_samples)

        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        x = self.bottleneck(x)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        logits = self.out(x)

        # Restore: batch x electrodes x time
        logits = logits.reshape(batch_size, electrodes, time_samples)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TemporalUNetNoChannelMixing()
    dummy = torch.randn(2, 59, 2500)
    out = model(dummy)
    print("input:", tuple(dummy.shape))
    print("output:", tuple(out.shape))
    print("parameters:", count_parameters(model))
    assert out.shape == dummy.shape

