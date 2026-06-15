"""Abstract base class for text detectors used in the benchmark."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import DetectionBox


class Detector(ABC):
    name: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectionBox]:
        """Return text-region predictions for one RGB uint8 image (H, W, 3)."""
