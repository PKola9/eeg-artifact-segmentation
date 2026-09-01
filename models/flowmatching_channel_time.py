"""
Version 2 flow-matching model for channel-time EEG windows.

This model is used for self-supervised pretraining before artifact segmentation.

Input:
    xt: batch x electrodes x time
    t:  batch

Output:
    predicted velocity: batch x electrodes x time

Flow-matching objective:
    x0 = Gaussian noise
    x1 = real EEG window
    xt = (1 - t) * x0 + t * x1
    target velocity = x1 - x0

The encoder/decoder block names match SplitUNetChannelTimeSegmenter so compatible
weights can be transferred into the supervised segmentation model.
"""

from __future__ import annotations

import torch
from torch import nn

from splitunet_channel_time_segmentation import DownTime, SplitConvBlock, UpTime


class FlowMatchingChannelTimeUNet(nn.Module):
    def __init__(self, base_features: int = 8):
        super().__init__()
        self.time_bias = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.down1 = DownTime(1, base_features)
        self.down2 = DownTime(base_features, base_features * 2)
        self.bottleneck = SplitConvBlock(base_features * 2, base_features * 4)
        self.up2 = UpTime(base_features * 4, base_features * 2, base_features * 2)
        self.up1 = UpTime(base_features * 2, base_features, base_features)
        self.out = nn.Conv2d(base_features, 1, kernel_size=1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # xt: batch x electrodes x time
        x = xt.unsqueeze(1)
        bias = self.time_bias(t.view(-1, 1)).view(-1, 1, 1, 1)
        x = x + bias

        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        x = self.bottleneck(x)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        velocity = self.out(x).squeeze(1)
        return velocity


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FlowMatchingChannelTimeUNet()
    x = torch.randn(2, 59, 2500)
    t = torch.rand(2)
    y = model(x, t)
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print("parameters:", count_parameters(model))
    assert y.shape == x.shape
