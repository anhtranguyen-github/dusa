"""Download FUNSD (Form Understanding in Noisy Scanned Documents) from HuggingFace.

Source: nielsr/funsd (149 train + 50 test = canonical 199 forms).

Word-level BIO scheme: 0=O, 1=B-HEADER, 2=I-HEADER, 3=B-QUESTION, 4=I-QUESTION,
                      5=B-ANSWER, 6=I-ANSWER.

We normalize each form to the same per-line list format used by SROIE so the same
loader code can read both datasets:

    [{"box": [x0,y0,x1,y1], "label": "<bio_tag>", "text": "<word>"}, ...]

Output:
  data/funsd/
    images/{train,test}/<id>.png      # grayscale form images
    annotations/{train,test}/<id>.json
    splits/{train,test}.txt

Usage:
    python scripts/download_funsd.py
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO = "nielsr/funsd"
LABEL_MAP = {
    0: "O",
    1: "B-HEADER", 2: "I-HEADER",
    3: "B-QUESTION", 4: "I-QUESTION",
    5: "B-ANSWER", 6: "I-ANSWER",
}

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "funsd"


def main() -> None:
    print(f"→ Loading {REPO} …")
    dsd = load_dataset(REPO)

    for split in ("train", "test"):
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "annotations" / split).mkdir(parents=True, exist_ok=True)
    (DATA / "splits").mkdir(parents=True, exist_ok=True)

    for split, ds in dsd.items():
        ids = []
        for ex in ds:
            rid = ex["id"]
            ids.append(rid)
            (DATA / "images" / split / f"{rid}.png").write_bytes(b"")  # placeholder
            ex["image"].save(DATA / "images" / split / f"{rid}.png", "PNG")

            lines = [
                {"box": list(map(int, bbox)),
                 "label": LABEL_MAP[tag],
                 "text": word}
                for word, bbox, tag in zip(ex["words"], ex["bboxes"], ex["ner_tags"])
            ]
            with open(DATA / "annotations" / split / f"{rid}.json", "w") as f:
                json.dump(lines, f, ensure_ascii=False)

        (DATA / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n")
        print(f"  {split}: {len(ids)} forms")

    print(f"✓ FUNSD ready under {DATA}")


if __name__ == "__main__":
    main()
