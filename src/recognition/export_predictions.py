"""Run a recognition adapter over SROIE test crops and emit per-image JSON.

Output schema (consumed by S3 LayoutLMv3 and S4 Qwen):

    {
      "<image_id>": [
        {"bbox": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
         "text": "...",
         "conf": null}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _axis_aligned_crop(image: np.ndarray, polygon: list) -> np.ndarray | None:
    xs = [int(round(p[0])) for p in polygon]
    ys = [int(round(p[1])) for p in polygon]
    x0, x1 = max(min(xs), 0), min(max(xs), image.shape[1])
    y0, y1 = max(min(ys), 0), min(max(ys), image.shape[0])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image[y0:y1, x0:x1]


def build_ocr_json(
    examples: Iterable[tuple[str, np.ndarray, list]],
    adapter,
    out_path: Path,
) -> int:
    """Crop, predict, group by image_id, write JSON. Returns total line count."""
    crops: list[np.ndarray] = []
    refs: list[tuple[str, int]] = []
    grouped: dict[str, list] = {}

    for image_id, image, lines in examples:
        grouped.setdefault(image_id, [])
        for ln in lines:
            polygon = ln["polygon"] if isinstance(ln, dict) else ln.polygon
            crop = _axis_aligned_crop(image, polygon)
            if crop is None:
                continue
            slot = len(grouped[image_id])
            grouped[image_id].append({"bbox": [[int(p[0]), int(p[1])] for p in polygon],
                                     "text": None, "conf": None})
            crops.append(crop)
            refs.append((image_id, slot))

    preds = adapter.predict(crops)
    for (image_id, slot), text in zip(refs, preds):
        grouped[image_id][slot]["text"] = text

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(grouped, indent=2, ensure_ascii=False))
    return len(refs)


def _iter_sroie_examples():
    from ..data.sroie_loader import iter_split
    for image_id, image, gt in iter_split("test"):
        lines = [{"polygon": box.polygon, "text": box.text} for box in gt]
        yield image_id, image, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Optional fine-tuned PARSeq .ckpt; default = pretrained.")
    ap.add_argument("--out", default="data/processed/recognition/ocr_predictions_test.json")
    args = ap.parse_args()

    from .adapters import PARSeqAdapter
    ckpt = args.checkpoint if args.checkpoint else None
    adapter = PARSeqAdapter(checkpoint_path=ckpt)
    n = build_ocr_json(_iter_sroie_examples(), adapter, ROOT / args.out)
    print(f"wrote {args.out}  ({n} lines)")


if __name__ == "__main__":
    main()
