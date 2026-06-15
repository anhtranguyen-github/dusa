"""Minimal CRNN (CNN + BiLSTM + CTC) used as a from-scratch baseline.

Trained briefly on SROIE crops if no clean pretrained CRNN loads from HF.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789.,:;()/-+*$%&'@!?# []"
)


class CRNNTiny(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((4, 1), (4, 1)),
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        b, c, h, t = feat.shape
        assert h == 1, f"expected H=1 after CNN, got {h}"
        feat = feat.squeeze(2).permute(2, 0, 1)
        out, _ = self.rnn(feat)
        return self.fc(out)


def ctc_greedy_decode(logits: torch.Tensor, charset: str) -> list[str]:
    """Greedy CTC decode. logits: (T, B, C). Blank index = len(charset)."""
    blank = len(charset)
    pred_ids = logits.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out: list[str] = []
    for row in pred_ids:
        chars: list[str] = []
        prev = -1
        for idx in row:
            if idx != prev and idx != blank:
                if 0 <= idx < len(charset):
                    chars.append(charset[idx])
            prev = idx
        out.append("".join(chars))
    return out
