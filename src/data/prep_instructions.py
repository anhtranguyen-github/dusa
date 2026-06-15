"""S4 — Build the SROIE instruction dataset for Qwen2.5-3B (zero-shot + QLoRA).

For each receipt, aggregate line text in reading order (top-to-bottom by line
y-center, tie-break by x-center) and assemble the 4 KIE fields by joining all
text whose line label matches.

Output:
  data/processed/instructions/
    sroie_kie_instructions_train.jsonl   # {messages: [...]}
    sroie_kie_instructions_test.jsonl
    sroie_kie_test_targets.json          # {<image_id>: {company, date, address, total}}
    prompt_template.md                   # documented prompt for reproducibility

Each train/test record uses the OpenAI/Qwen chat format ready for TRL SFTTrainer.

Usage:
    python -m src.data.prep_instructions
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .sroie_loader import iter_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "instructions"

SYSTEM = (
    "You are a strict information extractor. Given OCR text from a receipt, "
    "return a JSON object with exactly four fields: company, date, address, total. "
    "Use null when a field is not present. Output JSON only, no prose."
)

USER_TEMPLATE = (
    "Extract from this receipt text:\n"
    "```\n{ocr_text}\n```\n"
    'Output JSON: {{"company": ..., "date": ..., "address": ..., "total": ...}}'
)


def _line_text_in_reading_order(gt) -> str:
    items = []
    for box in gt:
        text = (box.text or "").strip()
        if not text:
            continue
        ys = [p[1] for p in box.polygon]
        xs = [p[0] for p in box.polygon]
        items.append(((sum(ys) / len(ys), sum(xs) / len(xs)), text, box.label or "other"))
    items.sort(key=lambda x: (round(x[0][0] / 8), x[0][1]))  # coarse y-bucket then x
    return "\n".join(t for _, t, _ in items), items


def _targets(items) -> dict:
    buckets: dict[str, list[str]] = defaultdict(list)
    for _, text, label in items:
        if label in ("company", "date", "address", "total"):
            buckets[label].append(text)
    return {f: " ".join(buckets[f]) if buckets[f] else None
            for f in ("company", "date", "address", "total")}


def prep_split(split: str) -> dict:
    out_path = OUT / f"sroie_kie_instructions_{split}.jsonl"
    n = 0
    targets: dict[str, dict] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for image_id, _image, gt in iter_split(split):
            ocr_text, items = _line_text_in_reading_order(gt)
            tgt = _targets(items)
            targets[image_id] = tgt
            record = {
                "id": image_id,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TEMPLATE.format(ocr_text=ocr_text)},
                    {"role": "assistant", "content": json.dumps(tgt, ensure_ascii=False)},
                ],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return out_path, n, targets


def main() -> None:
    argparse.ArgumentParser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prompt_template.md").write_text(
        "## Qwen2.5-3B SROIE KIE prompt\n\n"
        "**system:**\n```\n" + SYSTEM + "\n```\n\n"
        "**user:** rendered with the receipt's OCR text:\n```\n" + USER_TEMPLATE + "\n```\n\n"
        "**assistant:** JSON with keys `company`, `date`, `address`, `total` (or `null`).\n"
    )
    for split in ("train", "test"):
        path, n, targets = prep_split(split)
        print(f"  {split}: {n} examples → {path.relative_to(ROOT)}")
        if split == "test":
            (OUT / "sroie_kie_test_targets.json").write_text(
                json.dumps(targets, ensure_ascii=False, indent=2)
            )
    print(f"✓ wrote {OUT}")


if __name__ == "__main__":
    main()
