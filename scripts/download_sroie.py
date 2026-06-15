"""Download SROIE from HuggingFace and materialize on disk as image + JSON pairs.

Source: arvindrajan92/sroie_document_understanding (652 receipts).
Each example has:
  - image: PIL receipt image
  - ocr:   list of {box: [[x,y]*4], label: "company|date|address|total|other", text: str}

Output layout (matches the course's expected SROIE format):
  data/sroie/
    images/{train,test}/<id>.jpg
    annotations/{train,test}/<id>.json   ← list of line dicts
    splits/{train,test}.txt              ← image IDs (deterministic 80/20)

Usage:
    python scripts/download_sroie.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

REPO = "arvindrajan92/sroie_document_understanding"
SEED = 42
TEST_FRAC = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sroie"


def main() -> None:
    print(f"→ Loading {REPO} …")
    ds = load_dataset(REPO)["train"]
    n = len(ds)
    print(f"  got {n} examples")

    ids = [f"{i:04d}" for i in range(n)]
    rng = random.Random(SEED)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    n_test = int(n * TEST_FRAC)
    test_ids = set(shuffled[:n_test])

    for split in ("train", "test"):
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "annotations" / split).mkdir(parents=True, exist_ok=True)
    (DATA / "splits").mkdir(parents=True, exist_ok=True)

    train_list, test_list = [], []
    for i, ex in enumerate(ds):
        rid = f"{i:04d}"
        split = "test" if rid in test_ids else "train"
        (test_list if split == "test" else train_list).append(rid)

        img_path = DATA / "images" / split / f"{rid}.jpg"
        ann_path = DATA / "annotations" / split / f"{rid}.json"

        ex["image"].convert("RGB").save(img_path, "JPEG", quality=92)
        with open(ann_path, "w") as f:
            json.dump(ex["ocr"], f, ensure_ascii=False)

        if (i + 1) % 50 == 0:
            print(f"  wrote {i+1}/{n}")

    (DATA / "splits" / "train.txt").write_text("\n".join(train_list) + "\n")
    (DATA / "splits" / "test.txt").write_text("\n".join(test_list) + "\n")

    print(f"\n✓ done")
    print(f"  train: {len(train_list)} images → {DATA/'images'/'train'}")
    print(f"  test:  {len(test_list)} images → {DATA/'images'/'test'}")
    print(f"  splits → {DATA/'splits'}")


if __name__ == "__main__":
    main()
