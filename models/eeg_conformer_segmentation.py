"""
Lightweight EEG-Conformer-style baseline for dense artifact segmentation.

Purpose:
    This model is an adapted Transformer-style EEG baseline for the thesis.
    Published EEG Conformer models are usually designed for EEG classification,
    not channel-time artifact segmentation. Here we keep the key idea:

        local temporal convolution + channel mixing + Transformer attention

    but change the output head so the model predicts a dense artifact logit for
    every EEG channel and time point.

Input:
    batch x channels x time

Output:
    batch x channels x time logits

Important:
    This is a fair internal baseline because it is trained and evaluated on the
    exact same PhysioMotion train/validation/test split as the Split U-Net
    models. It should not be compared directly to published EEG Conformer
    classification accuracies, because the task and metrics are different.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EEGConformerSegmentation(nn.Module):
    """
    Compact CNN-Transformer model for channel-time EEG artifact segmentation.

    The model compresses time into patches, applies Transformer attention over
    those temporal patches, upsamples back to the original temporal resolution,
    and uses a 1x1 projection to produce one artifact logit per channel-time
    point.
    """

    def __init__(
        self,
        channels: int = 59,
        embed_dim: int = 64,
        patch_stride: int = 10,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.patch_stride = patch_stride

        # Local temporal features per EEG channel.
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(channels, embed_dim, kernel_size=25, padding=12),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=15, stride=patch_stride, padding=7),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attention_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.decoder = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, channels, kernel_size=1),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: batch x channels x time
        original_time = eeg.shape[-1]

        x = self.temporal_conv(eeg)  # batch x embed_dim x reduced_time
        x = x.transpose(1, 2)  # batch x reduced_time x embed_dim
        x = self.transformer(x)
        x = x.transpose(1, 2)  # batch x embed_dim x reduced_time

        x = F.interpolate(x, size=original_time, mode="linear", align_corners=False)
        logits = self.decoder(x)  # batch x channels x time
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EEGConformerSegmentation()
    dummy = torch.randn(2, 59, 2500)
    out = model(dummy)
    print("input:", tuple(dummy.shape))
    print("output:", tuple(out.shape))
    print("parameters:", count_parameters(model))
    assert out.shape == dummy.shape
