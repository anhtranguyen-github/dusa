"""Download CORD-v2 (Consolidated Receipt Dataset) from HuggingFace.

Source: naver-clova-ix/cord-v2 (800 train + 100 val + 100 test = official 1000 receipts).

Each example: image + ground_truth (Donut-style nested JSON with menu/sub_total/total/...).
We save the parsed gt_parse dict directly — no normalization since CORD's annotation
is hierarchical and lossy to flatten.

Output:
  data/cord/
    images/{train,val,test}/<id>.png
    annotations/{train,val,test}/<id>.json    # parsed ground_truth['gt_parse']
    splits/{train,val,test}.txt

Usage:
    python scripts/download_cord.py
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO = "naver-clova-ix/cord-v2"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cord"


def main() -> None:
    print(f"→ Loading {REPO} …")
    dsd = load_dataset(REPO)

    split_map = {"train": "train", "validation": "val", "test": "test"}
    for src, dst in split_map.items():
        (DATA / "images" / dst).mkdir(parents=True, exist_ok=True)
        (DATA / "annotations" / dst).mkdir(parents=True, exist_ok=True)
    (DATA / "splits").mkdir(parents=True, exist_ok=True)

    for src, dst in split_map.items():
        ds = dsd[src]
        ids = []
        for i, ex in enumerate(ds):
            rid = f"{i:04d}"
            ids.append(rid)
            ex["image"].save(DATA / "images" / dst / f"{rid}.png", "PNG", optimize=True)
            gt = json.loads(ex["ground_truth"])
            with open(DATA / "annotations" / dst / f"{rid}.json", "w") as f:
                json.dump(gt.get("gt_parse", gt), f, ensure_ascii=False)
            if (i + 1) % 100 == 0:
                print(f"  {dst}: {i+1}/{len(ds)}")
        (DATA / "splits" / f"{dst}.txt").write_text("\n".join(ids) + "\n")
        print(f"  {dst}: {len(ids)} receipts")

    print(f"✓ CORD-v2 ready under {DATA}")


if __name__ == "__main__":
    main()
