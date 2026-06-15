"""Detection package."""
from .base import Detector
from .doctr_backends import DBNetDetector, MixNetDetector
from .evaluation import AggregateStats, ImageStats, aggregate, evaluate_image, polygon_iou
from .types import DetectionBox, Polygon
from .visualization import draw_overlay

__all__ = [
    "AggregateStats",
    "DBNetDetector",
    "DetectionBox",
    "Detector",
    "ImageStats",
    "MixNetDetector",
    "Polygon",
    "aggregate",
    "draw_overlay",
    "evaluate_image",
    "polygon_iou",
]
