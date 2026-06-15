"""S2 — Crop word/line images from SROIE for PARSeq fine-tuning.

The Buổi 2 lab spec says: "Crop word images từ SROIE bbox (dùng output MixNet buổi 1)".
SROIE GT is line-level, so by default we produce **line crops** (the realistic
unit OCR'd by document detectors on receipts). When MixNet/PARSeq is later trained
on word-level boxes, swap `source=...` for a MixNet predictions JSON.

Crop strategy:
  1. Compute axis-aligned bbox from polygon (SROIE polygons are near-axis-aligned).
  2. Optional perspective warp (for skewed text) — disabled by default for SROIE.
  3. Skip crops with empty/whitespace-only text and crops smaller than 8 px in either dim.

Output:
  data/processed/recognition/
    {train,test}/images/<image_id>__<line_idx>.jpg
    {train,test}/labels.txt          # "<relative_path>\t<text>" per line
    splits.json                       # {n_train, n_test, skipped, sources}

Usage:
    python -m src.data.prep_recognition
    python -m src.data.prep_recognition --warp   # perspective-warp polygons
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .sroie_loader import iter_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "recognition"

MIN_SIDE = 8


def _axis_aligned_crop(image: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray | None:
    xs = [int(round(p[0])) for p in polygon]
    ys = [int(round(p[1])) for p in polygon]
    x0, x1 = max(min(xs), 0), min(max(xs), image.shape[1])
    y0, y1 = max(min(ys), 0), min(max(ys), image.shape[0])
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        return None
    return image[y0:y1, x0:x1]


def _perspective_warp(image: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray | None:
    pts = np.array(polygon, dtype=np.float32)
    if pts.shape != (4, 2):
        return None
    # Order: top-left, top-right, bottom-right, bottom-left.
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)
    w = int(max(np.linalg.norm(rect[1] - rect[0]),
                np.linalg.norm(rect[2] - rect[3])))
    h = int(max(np.linalg.norm(rect[2] - rect[1]),
                np.linalg.norm(rect[3] - rect[0])))
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


def prep_split(split: str, warp: bool) -> dict:
    out_img = OUT / split / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    label_lines: list[str] = []
    skipped_empty = skipped_small = 0
    n_kept = 0

    for image_id, image, gt in iter_split(split):
        for idx, box in enumerate(gt):
            text = (box.text or "").strip()
            if not text:
                skipped_empty += 1
                continue
            crop = _perspective_warp(image, box.polygon) if warp else _axis_aligned_crop(image, box.polygon)
            if crop is None:
                skipped_small += 1
                continue
            rel = f"images/{image_id}__{idx:03d}.jpg"
            cv2.imwrite(str(out_img / f"{image_id}__{idx:03d}.jpg"),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            label_lines.append(f"{rel}\t{text}")
            n_kept += 1

    (OUT / split / "labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return {"split": split, "kept": n_kept, "skipped_empty": skipped_empty, "skipped_small": skipped_small}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warp", action="store_true",
                    help="Perspective-warp polygons (set for skewed text). Default: axis-aligned crop.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"crop": "perspective_warp" if args.warp else "axis_aligned",
               "splits": []}
    for split in ("train", "test"):
        r = prep_split(split, warp=args.warp)
        summary["splits"].append(r)
        print(f"  {split}: kept={r['kept']}  empty={r['skipped_empty']}  small={r['skipped_small']}")

    (OUT / "splits.json").write_text(json.dumps(summary, indent=2))
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
