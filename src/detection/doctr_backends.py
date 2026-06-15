"""DocTR-backed text detectors used in the Session-1 benchmark.

We wrap two DocTR detectors so the benchmark can compare architectures with a
single interface:

  - DBNetDetector: DB-Net pretrained (the curriculum's baseline).
  - MixNetDetector: a placeholder for the to-be-fine-tuned MixNet. Until
    `mixnet_sroie_finetuned.pth` exists, it falls back to DocTR's FAST detector
    (a different real-time text detector) so the benchmark runs end-to-end and
    can be re-run trivially once real MixNet weights are dropped in.

DocTR predictor output: list[dict] where dict['words'] is an ndarray of shape
(N, 5, 2). The first 4 points are the polygon corners (normalized 0-1), and
the 5th point is [0, score]. We denormalize to pixel coords here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from doctr.models import detection_predictor

from .base import Detector
from .types import DetectionBox


def _docs_to_boxes(raw: np.ndarray, image_shape: tuple[int, int]) -> list[DetectionBox]:
    """raw: (N, 5, 2) from DocTR; image_shape: (H, W). Returns DetectionBoxes in pixel space."""
    H, W = image_shape
    boxes: list[DetectionBox] = []
    for row in raw:
        polygon = [(float(x) * W, float(y) * H) for x, y in row[:4]]
        score = float(row[4, 1])
        boxes.append(DetectionBox(polygon=polygon, score=score))
    return boxes


class DBNetDetector(Detector):
    """DocTR DB-Net (ResNet-50 backbone, pretrained). Buổi 1 baseline."""

    name = "dbnet_pretrained"

    def __init__(self, arch: str = "db_resnet50", pretrained: bool = True) -> None:
        self._pred = detection_predictor(
            arch=arch,
            pretrained=pretrained,
            assume_straight_pages=False,
            preserve_aspect_ratio=True,
            symmetric_pad=True,
        )

    def detect(self, image: np.ndarray) -> list[DetectionBox]:
        out = self._pred([image])[0]
        raw = out["words"] if isinstance(out, dict) else out
        return _docs_to_boxes(np.asarray(raw), image.shape[:2])


class MixNetDetector(Detector):
    """Placeholder for fine-tuned MixNet.

    When `checkpoint` exists, weights are loaded via the project's MixNet impl
    (TODO: add in src/detection/mixnet_arch.py once the official repo is vendored).
    Until then, we use DocTR's FAST detector so the benchmark scaffold runs and
    produces a usable second column in the comparison table.
    """

    name = "mixnet_finetuned"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        fallback_arch: str = "fast_base",
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        if self.checkpoint and self.checkpoint.exists():
            raise NotImplementedError(
                "MixNet checkpoint loading is not implemented yet. "
                "Vendor the MixNet architecture under src/detection/mixnet_arch.py "
                "and load weights here. See Session 1 lab for the fine-tune pipeline."
            )
        # Fallback so the benchmark still produces a comparison column.
        self._using_fallback = True
        self._pred = detection_predictor(
            arch=fallback_arch,
            pretrained=True,
            assume_straight_pages=False,
            preserve_aspect_ratio=True,
            symmetric_pad=True,
        )

    def detect(self, image: np.ndarray) -> list[DetectionBox]:
        out = self._pred([image])[0]
        raw = out["words"] if isinstance(out, dict) else out
        return _docs_to_boxes(np.asarray(raw), image.shape[:2])
