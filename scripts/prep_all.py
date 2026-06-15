"""Run every SROIE data-prep step in curriculum order.

Detection (S1) consumes the raw `data/sroie/annotations/` directly — no prep
needed. The other three sessions produce derivative artifacts under
`data/processed/`.

Usage:
    python scripts/prep_all.py
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEPS = [
    ("S2 recognition crops",     "src.data.prep_recognition"),
    ("S3 KIE BIO/bbox alignment", "src.data.prep_kie"),
    ("S4 Qwen instruction JSONL", "src.data.prep_instructions"),
]


def main() -> None:
    for label, module in STEPS:
        print(f"\n=== {label} ({module}) ===")
        t0 = time.time()
        mod = importlib.import_module(module)
        mod.main()
        print(f"  {time.time() - t0:.1f}s")

    print("\nArtifacts:")
    root = Path("data/processed")
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.suffix.lower() in {".jpg", ".png"}:
                print(f"  {p}")


if __name__ == "__main__":
    main()
