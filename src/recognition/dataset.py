"""PyTorch Dataset for SROIE recognition crops.

Reads the labels.txt produced by `src/data/prep_recognition.py`:
    images/<id>__<line_idx>.jpg \t <text>

Returns (image_tensor, text_string). The Trainer-side collator should
handle the char-tokenization (PARSeq uses its own tokenizer that comes with
the model checkpoint).
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _load_labels(root: Path, labels_rel: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    with open(root / labels_rel, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                continue
            rel_img, text = line.split("\t", 1)
            items.append((rel_img, text))
    return items


class SROIERecognitionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        labels_rel: str,
        image_height: int = 32,
        image_width: int = 128,
        augment: bool = False,
        max_label_length: int = 32,
    ) -> None:
        self.root = Path(root)
        self.items = _load_labels(self.root, labels_rel)
        self.h = image_height
        self.w = image_width
        self.max_len = max_label_length

        tf: list[torch.nn.Module] = [
            transforms.Resize((self.h, self.w), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        if augment:
            tf.insert(0, transforms.ColorJitter(0.2, 0.2, 0.2))
            tf.insert(0, transforms.RandomAffine(degrees=5, shear=2))
        self.tf = transforms.Compose(tf)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        rel_img, text = self.items[idx]
        img = Image.open(self.root / Path(*rel_img.split("/"))).convert("RGB")
        text = text[: self.max_len]
        return self.tf(img), text
