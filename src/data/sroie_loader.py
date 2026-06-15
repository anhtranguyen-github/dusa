"""SROIE loader for the detection benchmark.

After `scripts/download_sroie.py`, each `data/sroie/annotations/<split>/<id>.json`
is a list of:
    {"box": [[x0,y0], [x1,y1], [x2,y2], [x3,y3]],
     "label": "company|date|address|total|other",
     "text": "..."}

This loader yields (image_id, np.ndarray RGB image, list[DetectionBox]) so
detection benchmarks can iterate cleanly. Labels are preserved on the GT boxes
for downstream KIE work, but the detection benchmark itself only uses the
polygons (any text region is a valid detection target).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import json
import numpy as np
from PIL import Image

from ..detection.types import DetectionBox, Polygon


def _root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "sroie"


def list_ids(split: str = "test") -> list[str]:
    ids = (_root() / "splits" / f"{split}.txt").read_text().splitlines()
    return [i for i in ids if i.strip()]


def _to_polygon(box: list) -> Polygon:
    # SROIE bbox is [[x,y]*4]; numbers may be float.
    return [(float(x), float(y)) for x, y in box]


def load_example(image_id: str, split: str = "test") -> tuple[np.ndarray, list[DetectionBox]]:
    img_path = _root() / "images" / split / f"{image_id}.jpg"
    ann_path = _root() / "annotations" / split / f"{image_id}.json"
    image = np.array(Image.open(img_path).convert("RGB"))
    with open(ann_path) as f:
        raw = json.load(f)
    boxes = [
        DetectionBox(polygon=_to_polygon(line["box"]),
                     score=1.0,
                     text=line.get("text"),
                     label=line.get("label"))
        for line in raw
    ]
    return image, boxes


def iter_split(
    split: str = "test",
    limit: int | None = None,
) -> Iterator[tuple[str, np.ndarray, list[DetectionBox]]]:
    ids = list_ids(split)
    if limit is not None:
        ids = ids[:limit]
    for rid in ids:
        image, gt = load_example(rid, split=split)
        yield rid, image, gt
