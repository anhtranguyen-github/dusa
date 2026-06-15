"""HuggingFace Dataset wrapper for the SROIE KIE jsonl produced by prep_kie.py.

Reads {id, image_path, image_size, words, boxes, ner_tags} per line. The
LayoutLMv3Processor handles WordPiece tokenization + bbox→[0,1000] normalization
+ image preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from datasets import Dataset


def load_split(prepared_dir: Path, jsonl_name: str) -> Dataset:
    records: list[dict] = []
    with open(prepared_dir / jsonl_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def load_image(image_path: str | Path, project_root: Path) -> Image.Image:
    p = (project_root / image_path).resolve() if not Path(image_path).is_absolute() else Path(image_path)
    return Image.open(p).convert("RGB")
