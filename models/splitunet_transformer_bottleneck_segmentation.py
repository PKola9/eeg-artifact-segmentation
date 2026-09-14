"""
Split U-Net with an attention bottleneck for dense EEG artifact segmentation.

This model is the post-thesis "future work" implementation:

    flow-initialized Split U-Net + Transformer bottleneck

Why this design:
    - The Split U-Net keeps the dense channel-time segmentation structure.
    - The CNN blocks still match the flow-matching checkpoint, so the learned
      flow-matching weights can initialize the convolutional encoder/decoder.
    - The Transformer bottleneck adds longer-range temporal context after the
      signal has been compressed in time, making attention computationally
      manageable.

Input:
    batch x electrodes x time

Output:
    batch x electrodes x time logits
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from .splitunet_channel_time_segmentation import DownTime, SplitConvBlock, UpTime
except ImportError:
    from splitunet_channel_time_segmentation import DownTime, SplitConvBlock, UpTime


class TemporalTransformerBottleneck(nn.Module):
    """
    Lightweight temporal attention block used inside the U-Net bottleneck.

    The bottleneck feature map has shape:
        batch x features x electrodes x compressed_time

    Full attention over every electrode-time point would be very expensive.
    Instead, this block averages across electrodes to get one temporal token
    sequence, applies Transformer attention over time, and broadcasts the
    temporal context back to all electrodes.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch x channels x electrodes x compressed_time
        temporal_tokens = x.mean(dim=2).transpose(1, 2)
        temporal_context = self.encoder(temporal_tokens)
        temporal_context = self.norm(temporal_context)
        temporal_context = temporal_context.transpose(1, 2).unsqueeze(2)
        return x + temporal_context


class SplitUNetTransformerBottleneckSegmenter(nn.Module):
    """
    Dense channel-time segmentation model with a Transformer bottleneck.

    Compatible convolutional module names are intentionally kept identical to
    SplitUNetChannelTimeSegmenter:
        down1, down2, bottleneck, up2, up1

    This allows the existing flow-matching weight-transfer function to copy the
    matching CNN weights. The new attention block is trained from scratch.
    """

    def __init__(
        self,
        base_features: int = 8,
        attention_heads: int = 4,
        attention_layers: int = 1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        bottleneck_features = base_features * 4

        self.down1 = DownTime(1, base_features)
        self.down2 = DownTime(base_features, base_features * 2)
        self.bottleneck = SplitConvBlock(base_features * 2, bottleneck_features)
        self.attention_bottleneck = TemporalTransformerBottleneck(
            channels=bottleneck_features,
            num_heads=attention_heads,
            num_layers=attention_layers,
            dropout=attention_dropout,
        )
        self.up2 = UpTime(bottleneck_features, base_features * 2, base_features * 2)
        self.up1 = UpTime(base_features * 2, base_features, base_features)
        self.out = nn.Conv2d(base_features, 1, kernel_size=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: batch x electrodes x time
        x = eeg.unsqueeze(1)

        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        x = self.bottleneck(x)
        x = self.attention_bottleneck(x)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        logits = self.out(x)
        return logits.squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SplitUNetTransformerBottleneckSegmenter()
    dummy = torch.randn(2, 59, 2500)
    out = model(dummy)
    print("input:", tuple(dummy.shape))
    print("output:", tuple(out.shape))
    print("parameters:", count_parameters(model))
    assert out.shape == dummy.shape
