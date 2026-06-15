"""Polygon-IoU-based detection evaluation (precision / recall / F1).

Matches each predicted box to at most one ground-truth box at IoU ≥ threshold,
using greedy assignment by descending prediction score. Returns per-image counts
and aggregate (micro-averaged) precision / recall / F1.
"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from .types import DetectionBox


@dataclass
class ImageStats:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class AggregateStats:
    tp: int
    fp: int
    fn: int
    n_images: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _safe_polygon(coords: list[tuple[float, float]]) -> ShapelyPolygon:
    poly = ShapelyPolygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
        # make_valid may return a MultiPolygon / GeometryCollection; take the largest piece.
        if hasattr(poly, "geoms"):
            poly = max(poly.geoms, key=lambda g: getattr(g, "area", 0.0))
    return poly


def polygon_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    pa, pb = _safe_polygon(a), _safe_polygon(b)
    if pa.area == 0 or pb.area == 0:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.union(pb).area
    return inter / union if union else 0.0


def evaluate_image(
    preds: list[DetectionBox],
    gts: list[DetectionBox],
    iou_thresh: float = 0.5,
) -> ImageStats:
    """Greedy match preds (sorted by score desc) to gts at IoU ≥ thresh."""
    matched_gt: set[int] = set()
    tp = 0

    sorted_preds = sorted(preds, key=lambda b: b.score, reverse=True)
    for pred in sorted_preds:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gts):
            if j in matched_gt:
                continue
            iou = polygon_iou(pred.polygon, gt.polygon)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt.add(best_j)
            tp += 1

    fp = len(sorted_preds) - tp
    fn = len(gts) - len(matched_gt)
    return ImageStats(tp=tp, fp=fp, fn=fn)


def aggregate(per_image: list[ImageStats]) -> AggregateStats:
    return AggregateStats(
        tp=sum(s.tp for s in per_image),
        fp=sum(s.fp for s in per_image),
        fn=sum(s.fn for s in per_image),
        n_images=len(per_image),
    )
