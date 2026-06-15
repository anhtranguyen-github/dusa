"""Driver script used by notebook 01 (and runnable standalone).

Runs DB-Net and MixNet (placeholder) on the first `--limit` SROIE test images,
computes P/R/F1 at IoU=0.5 (and 0.3 for the granularity caveat), saves:

  reports/benchmarks/detection_results.json
  reports/error_analysis/detection.md
  reports/benchmarks/detection_overlays/<id>_<detector>.png  (first 10)

Usage:
    python -m src.detection.run_benchmark --limit 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.data.sroie_loader import iter_split
from src.detection import (
    AggregateStats,
    DBNetDetector,
    DetectionBox,
    Detector,
    MixNetDetector,
    aggregate,
    draw_overlay,
    evaluate_image,
)

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "reports" / "benchmarks"
OVL_DIR = BENCH_DIR / "detection_overlays"
REPORT_PATH = ROOT / "reports" / "error_analysis" / "detection.md"


def run_one(
    detector: Detector,
    limit: int,
    iou_thresholds: tuple[float, ...] = (0.5, 0.3),
) -> dict:
    per_image_05: list = []
    per_image_03: list = []
    per_image_records: list[dict] = []
    overlays: list[tuple[str, np.ndarray, list[DetectionBox], list[DetectionBox]]] = []

    t0 = time.time()
    for rid, img, gt in iter_split("test", limit=limit):
        preds = detector.detect(img)
        s05 = evaluate_image(preds, gt, iou_thresh=0.5)
        s03 = evaluate_image(preds, gt, iou_thresh=0.3)
        per_image_05.append(s05)
        per_image_03.append(s03)
        per_image_records.append({
            "id": rid,
            "n_gt": len(gt),
            "n_pred": len(preds),
            "iou_0.5": {"tp": s05.tp, "fp": s05.fp, "fn": s05.fn,
                        "P": s05.precision, "R": s05.recall, "F1": s05.f1},
            "iou_0.3": {"tp": s03.tp, "fp": s03.fp, "fn": s03.fn,
                        "P": s03.precision, "R": s03.recall, "F1": s03.f1},
        })
        if len(overlays) < 10:
            overlays.append((rid, img, preds, gt))

    elapsed = time.time() - t0
    agg05 = aggregate(per_image_05)
    agg03 = aggregate(per_image_03)

    OVL_DIR.mkdir(parents=True, exist_ok=True)
    for rid, img, preds, gt in overlays:
        ovl = draw_overlay(img, preds, gt)
        cv2.imwrite(
            str(OVL_DIR / f"{rid}_{detector.name}.png"),
            cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR),
        )

    return {
        "detector": detector.name,
        "n_images": agg05.n_images,
        "elapsed_seconds": round(elapsed, 2),
        "fps": round(agg05.n_images / elapsed, 2) if elapsed else 0.0,
        "iou_0.5": {"P": round(agg05.precision, 4),
                    "R": round(agg05.recall, 4),
                    "F1": round(agg05.f1, 4),
                    "tp": agg05.tp, "fp": agg05.fp, "fn": agg05.fn},
        "iou_0.3": {"P": round(agg03.precision, 4),
                    "R": round(agg03.recall, 4),
                    "F1": round(agg03.f1, 4),
                    "tp": agg03.tp, "fp": agg03.fp, "fn": agg03.fn},
        "per_image": per_image_records,
    }


def write_report(results: list[dict], limit: int) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Detection benchmark — Session 1",
        "",
        f"Test set: first **{limit} SROIE** images "
        f"(`data/sroie/images/test/`).  IoU threshold matched greedily per image, "
        f"micro-averaged P / R / F1.",
        "",
        "## Results",
        "",
        "| Detector | IoU | Precision | Recall | F1 | TP | FP | FN | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        for iou in ("0.5", "0.3"):
            m = r[f"iou_{iou}"]
            lines.append(
                f"| `{r['detector']}` | {iou} | {m['P']:.3f} | {m['R']:.3f} | "
                f"{m['F1']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {r['fps']:.2f} |"
            )
    lines += [
        "",
        "## Granularity caveat",
        "",
        "SROIE ground truth is **line-level** (one polygon per receipt line), but "
        "DocTR's DB-Net / FAST predict **word-level** boxes. This systematically "
        "depresses precision (many small predictions per GT line) and IoU "
        "(word ∩ line / word ∪ line ≪ 1). The IoU=0.3 row is therefore more "
        "informative for relative ordering than IoU=0.5.",
        "",
        "For a fair like-for-like comparison, the follow-up step is either:",
        "  1. Merge predicted word boxes into lines via horizontal clustering "
        "(sort by y, then group by x-overlap), or",
        "  2. Fine-tune the detector on SROIE's line-level boxes — exactly what "
        "Buổi 1's MixNet lab does.",
        "",
        "## Error modes observed",
        "",
        "- Word-level over-segmentation of long line items (price, total).",
        "- Misses on faint thermal-printed text near receipt edges.",
        "- Confusion between dot-leader sequences (`. . . . .`) and short tokens.",
        "",
        "## Files",
        "",
        f"- Per-image and aggregate JSON: `{BENCH_DIR.relative_to(ROOT)}/detection_results.json`",
        f"- Sample overlays (10/detector): `{OVL_DIR.relative_to(ROOT)}/<id>_<detector>.png` "
        "(GT green, predictions orange)",
        "",
        "## Repro",
        "",
        "```",
        f"python -m src.detection.run_benchmark --limit {limit}",
        "```",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50,
                    help="Number of test images (curriculum default: 50).")
    args = ap.parse_args()

    detectors: list[Detector] = [DBNetDetector(), MixNetDetector()]
    results = []
    for d in detectors:
        print(f"\n→ {d.name}")
        r = run_one(d, limit=args.limit)
        m05 = r["iou_0.5"]
        m03 = r["iou_0.3"]
        print(f"  IoU=0.5  P={m05['P']:.3f} R={m05['R']:.3f} F1={m05['F1']:.3f}")
        print(f"  IoU=0.3  P={m03['P']:.3f} R={m03['R']:.3f} F1={m03['F1']:.3f}")
        print(f"  {r['fps']:.2f} fps over {r['n_images']} imgs ({r['elapsed_seconds']}s)")
        results.append(r)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / "detection_results.json"
    with open(out, "w") as f:
        json.dump({"limit": args.limit, "detectors": results}, f, indent=2)
    print(f"\n✓ {out}")

    write_report(results, args.limit)
    print(f"✓ {REPORT_PATH}")
    print(f"✓ overlays → {OVL_DIR}")


if __name__ == "__main__":
    main()
