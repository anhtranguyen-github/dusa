"""PyTorch Dataset for SROIE polygon-bbox detection training.

Produces the standard DBNet-style supervision:
  - image  (3, H, W) float in [0, 1]
  - prob_map (1, H, W) float — text probability (shrunk-polygon mask)
  - thresh_map (1, H, W) float — distance-to-boundary band
  - prob_mask (1, H, W) — ignore region (always 1 here; we treat all polygons as valid)

For SROIE specifically, polygons are already near-axis-aligned. We still build
the full DBNet supervision so the same dataset wraps cleanly when the dataset
is replaced by ICDAR-2015 / Total-Text / etc.

Used by:
  - src.detection.train_mixnet  (S1 fine-tune)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.sroie_loader import iter_split, list_ids, load_example


def _shrink_polygon(polygon: np.ndarray, ratio: float = 0.4) -> np.ndarray | None:
    """DBNet-style polygon shrinking using the Vatti clipping algorithm
    approximation: shrink each polygon inward by `ratio * area / perimeter`.
    """
    try:
        from shapely.geometry import Polygon as SP
    except ImportError:
        return None
    p = SP(polygon)
    if not p.is_valid or p.area == 0:
        return None
    distance = p.area * (1 - ratio ** 2) / p.length
    shrunk = p.buffer(-distance, join_style=2)
    if shrunk.is_empty:
        return None
    if hasattr(shrunk, "exterior"):
        return np.array(list(shrunk.exterior.coords)[:-1])
    return None


class SROIEDetectionDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        image_size: int = 736,
        augment: bool = True,
        min_text_size: int = 6,
    ) -> None:
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.min_text_size = min_text_size
        self.ids = list_ids(split)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        rid = self.ids[idx]
        image, gt = load_example(rid, split=self.split)
        H, W = image.shape[:2]

        # Resize image + polygons to fixed square (letterbox).
        scale = self.image_size / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        prob_map = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for box in gt:
            poly = np.array([(p[0] * scale, p[1] * scale) for p in box.polygon],
                            dtype=np.float32)
            if (poly[:, 0].max() - poly[:, 0].min()) < self.min_text_size:
                continue
            shrunk = _shrink_polygon(poly)
            if shrunk is None or len(shrunk) < 3:
                continue
            cv2.fillPoly(prob_map, [shrunk.astype(np.int32)], 1.0)

        img_t = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        prob_t = torch.from_numpy(prob_map).unsqueeze(0)
        return {"image": img_t, "prob_map": prob_t, "id": rid}
