"""
Version 2 SplitUNet model for channel-time EEG artifact segmentation.

Input:
    batch x electrodes x time

Output:
    batch x electrodes x time logits

After sigmoid:
    artifact probability for every EEG channel and every time sample

This is the professor-requested dense output form:
    input  = 59 x 2500
    output = 59 x 2500
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SplitConvBlock(nn.Module):
    """
    Factorized EEG block:
        temporal convolution -> electrode convolution

    Temporal convolution learns patterns along time.
    Electrode convolution learns relationships across EEG channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 9,
        electrode_kernel: int = 7,
    ):
        super().__init__()
        temporal_padding = temporal_kernel // 2
        electrode_padding = electrode_kernel // 2

        self.temporal = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_padding),
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.electrode = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(electrode_kernel, 1),
                padding=(electrode_padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.electrode(x)
        return x


class DownTime(nn.Module):
    """Downsample only along time, preserving electrode dimension."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = SplitConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.block(x)
        pooled = self.pool(features)
        return features, pooled


class UpTime(nn.Module):
    """Upsample only along time and combine with skip features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = SplitConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(x.shape[-2], skip.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SplitUNetChannelTimeSegmenter(nn.Module):
    """
    Predict artifact probability for every channel-time point.

    Difference from Version 1:
        Version 1 collapsed electrodes at the end and returned batch x time.
        Version 2 preserves electrodes and returns batch x electrodes x time.
    """

    def __init__(self, base_features: int = 8):
        super().__init__()
        self.down1 = DownTime(1, base_features)
        self.down2 = DownTime(base_features, base_features * 2)
        self.bottleneck = SplitConvBlock(base_features * 2, base_features * 4)
        self.up2 = UpTime(base_features * 4, base_features * 2, base_features * 2)
        self.up1 = UpTime(base_features * 2, base_features, base_features)
        self.out = nn.Conv2d(base_features, 1, kernel_size=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: batch x electrodes x time
        x = eeg.unsqueeze(1)  # batch x 1 x electrodes x time

        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        x = self.bottleneck(x)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        # batch x 1 x electrodes x time
        logits = self.out(x)

        # Version 2: keep electrode dimension.
        # batch x electrodes x time
        logits = logits.squeeze(1)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SplitUNetChannelTimeSegmenter()
    dummy = torch.randn(2, 59, 2500)
    out = model(dummy)
    print("input:", tuple(dummy.shape))
    print("output:", tuple(out.shape))
    print("parameters:", count_parameters(model))
    assert out.shape == dummy.shape
