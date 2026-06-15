"""Render predicted + ground-truth polygons over a source image for inspection."""
from __future__ import annotations

import cv2
import numpy as np

from .types import DetectionBox

GT_COLOR = (0, 200, 0)        # green
PRED_COLOR = (0, 80, 255)     # orange (RGB)


def draw_overlay(
    image: np.ndarray,
    preds: list[DetectionBox],
    gts: list[DetectionBox] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Return a copy of `image` with GT (green) and predictions (orange) overlaid."""
    canvas = image.copy()
    if gts:
        for gt in gts:
            pts = np.array(gt.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=GT_COLOR, thickness=thickness)
    for pred in preds:
        pts = np.array(pred.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=PRED_COLOR, thickness=thickness)
    return canvas
