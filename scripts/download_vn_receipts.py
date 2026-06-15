"""Download VN Receipts (MC-OCR 2021) from HuggingFace.

Source: DThai/mcocr-test (1199 train + 289 val + 296 test = 1784 receipts).

Note: despite the "-test" repo suffix, this is the full MC-OCR mirror with embedded
images and full Donut-style ground_truth JSON. Per-receipt fields are restaurant-
specific (Số hoá đơn, Ngày, Tổng cộng, etc.) — schema varies, so we keep the raw
parsed gt_parse dict.

Output:
  data/vn_receipts/
    images/{train,val,test}/<id>.jpg
    annotations/{train,val,test}/<id>.json    # parsed ground_truth['gt_parse']
    splits/{train,val,test}.txt

Usage:
    python scripts/download_vn_receipts.py
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO = "DThai/mcocr-test"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vn_receipts"


def main() -> None:
    print(f"→ Loading {REPO} …")
    dsd = load_dataset(REPO)

    split_map = {"train": "train", "validation": "val", "test": "test"}
    for dst in split_map.values():
        (DATA / "images" / dst).mkdir(parents=True, exist_ok=True)
        (DATA / "annotations" / dst).mkdir(parents=True, exist_ok=True)
    (DATA / "splits").mkdir(parents=True, exist_ok=True)

    for src, dst in split_map.items():
        ds = dsd[src]
        ids = []
        for i, ex in enumerate(ds):
            rid = f"{i:04d}"
            ids.append(rid)
            ex["image"].convert("RGB").save(
                DATA / "images" / dst / f"{rid}.jpg", "JPEG", quality=90,
            )
            gt = json.loads(ex["ground_truth"])
            with open(DATA / "annotations" / dst / f"{rid}.json", "w") as f:
                json.dump(gt.get("gt_parse", gt), f, ensure_ascii=False)
            if (i + 1) % 200 == 0:
                print(f"  {dst}: {i+1}/{len(ds)}")
        (DATA / "splits" / f"{dst}.txt").write_text("\n".join(ids) + "\n")
        print(f"  {dst}: {len(ids)} receipts")

    print(f"✓ VN Receipts (MC-OCR) ready under {DATA}")


if __name__ == "__main__":
    main()
