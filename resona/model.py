"""WhistleNet: a compact CNN over log-mel spectrogram windows (~95k parameters).

Small enough for real-time CPU inference; the same architecture could
later be quantised for a microcontroller.
"""
import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class WhistleNet(nn.Module):
    def __init__(self, n_mels: int = 64):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, frames, n_mels) -> logits (batch, 1)."""
        return self.head(self.features(x))
