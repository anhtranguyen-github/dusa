"""S3 — Prepare SROIE for LayoutLMv3 KIE fine-tuning.

For each receipt, emit one example with:
  - words:       per-word strings (split from each line by whitespace)
  - boxes:       per-word axis-aligned bbox in image coords [x0,y0,x1,y1]
  - ner_tags:    per-word BIO tag id over {O, B-COMPANY, I-COMPANY, B-DATE, I-DATE,
                                            B-ADDRESS, I-ADDRESS, B-TOTAL, I-TOTAL}
  - image_path:  relative path to the source receipt image
  - image_size:  [width, height]

Word bboxes are estimated by linear interpolation across the line polygon's width
(SROIE polygons are near axis-aligned). This is the same heuristic used by
microsoft/LayoutLMv3 reference scripts.

The processor (LayoutLMv3Processor) normalizes boxes to [0,1000] at training
time, so we keep raw pixel coords here.

Output:
  data/processed/kie/
    {train,test}.jsonl           # one json per receipt
    label_list.json              # ["O", "B-COMPANY", ...]

Usage:
    python -m src.data.prep_kie
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sroie_loader import iter_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "kie"

FIELDS = ("COMPANY", "DATE", "ADDRESS", "TOTAL")
LABEL_LIST = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}


def _aabb(polygon: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _word_boxes(line_bbox: tuple[int, int, int, int], words: list[str]) -> list[list[int]]:
    """Distribute a line's bbox across its words proportionally by character length."""
    x0, y0, x1, y1 = line_bbox
    if not words:
        return []
    weights = [max(len(w), 1) for w in words]
    total = sum(weights)
    width = x1 - x0
    boxes: list[list[int]] = []
    cursor = x0
    for w in weights:
        wpx = int(round(width * (w / total)))
        boxes.append([cursor, y0, cursor + wpx, y1])
        cursor += wpx
    # snap last word to line end (rounding drift)
    if boxes:
        boxes[-1][2] = x1
    return boxes


def _bio_for_line(label: str, n_words: int) -> list[str]:
    if not n_words:
        return []
    field = label.upper()
    if field not in FIELDS:
        return ["O"] * n_words
    tags = [f"B-{field}"] + [f"I-{field}"] * (n_words - 1)
    return tags


def _img_size(image_id: str, split: str) -> tuple[int, int]:
    from PIL import Image
    p = ROOT / "data" / "sroie" / "images" / split / f"{image_id}.jpg"
    with Image.open(p) as im:
        return im.size  # (width, height)


def prep_split(split: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{split}.jsonl"
    n_examples = n_words_total = n_tagged_total = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for image_id, _image, gt in iter_split(split):
            words: list[str] = []
            boxes: list[list[int]] = []
            tags: list[int] = []
            for box in gt:
                text = (box.text or "").strip()
                if not text:
                    continue
                w_list = text.split()
                line_bbox = _aabb(box.polygon)
                w_boxes = _word_boxes(line_bbox, w_list)
                bio = _bio_for_line(box.label or "other", len(w_list))
                for w, b, t in zip(w_list, w_boxes, bio):
                    words.append(w)
                    boxes.append(b)
                    tags.append(LABEL2ID[t])
                    if t != "O":
                        n_tagged_total += 1
            if not words:
                continue
            size = _img_size(image_id, split)
            fh.write(json.dumps({
                "id": image_id,
                "image_path": f"data/sroie/images/{split}/{image_id}.jpg",
                "image_size": list(size),
                "words": words,
                "boxes": boxes,
                "ner_tags": tags,
            }, ensure_ascii=False) + "\n")
            n_examples += 1
            n_words_total += len(words)
    return {"split": split, "examples": n_examples,
            "words": n_words_total, "tagged": n_tagged_total,
            "tag_rate": round(n_tagged_total / max(n_words_total, 1), 3)}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_list.json").write_text(json.dumps(LABEL_LIST, indent=2))
    for split in ("train", "test"):
        r = prep_split(split)
        print(f"  {r['split']}: {r['examples']} receipts, {r['words']} words, "
              f"{r['tagged']} tagged ({r['tag_rate']*100:.1f}%)")
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
