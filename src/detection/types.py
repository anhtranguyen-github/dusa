"""Shared types for detection benchmarking."""
from __future__ import annotations

from dataclasses import dataclass


Polygon = list[tuple[float, float]]  # 4 corners, clockwise from top-left


@dataclass(frozen=True)
class DetectionBox:
    """A single text region prediction (or ground truth)."""
    polygon: Polygon
    score: float = 1.0
    text: str | None = None
    label: str | None = None

    def to_xyxy(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))
