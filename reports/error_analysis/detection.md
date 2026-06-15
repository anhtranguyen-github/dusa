# Detection benchmark — Session 1

Test set: first **50 SROIE** images (`data/sroie/images/test/`).  IoU threshold matched greedily per image, micro-averaged P / R / F1.

## Results

| Detector | IoU | Precision | Recall | F1 | TP | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dbnet_pretrained` | 0.5 | 0.304 | 0.631 | 0.410 | 1686 | 3858 | 985 | 0.33 |
| `dbnet_pretrained` | 0.3 | 0.412 | 0.856 | 0.556 | 2285 | 3259 | 386 | 0.33 |
| `mixnet_finetuned` | 0.5 | 0.297 | 0.618 | 0.402 | 1651 | 3901 | 1020 | 0.43 |
| `mixnet_finetuned` | 0.3 | 0.409 | 0.850 | 0.552 | 2271 | 3281 | 400 | 0.43 |

## Granularity caveat

SROIE ground truth is **line-level** (one polygon per receipt line), but DocTR's DB-Net / FAST predict **word-level** boxes. This systematically depresses precision (many small predictions per GT line) and IoU (word ∩ line / word ∪ line ≪ 1). The IoU=0.3 row is therefore more informative for relative ordering than IoU=0.5.

For a fair like-for-like comparison, the follow-up step is either:
  1. Merge predicted word boxes into lines via horizontal clustering (sort by y, then group by x-overlap), or
  2. Fine-tune the detector on SROIE's line-level boxes — exactly what Buổi 1's MixNet lab does.

## Error modes observed

- Word-level over-segmentation of long line items (price, total).
- Misses on faint thermal-printed text near receipt edges.
- Confusion between dot-leader sequences (`. . . . .`) and short tokens.

## Files

- Per-image and aggregate JSON: `reports/benchmarks/detection_results.json`
- Sample overlays (10/detector): `reports/benchmarks/detection_overlays/<id>_<detector>.png` (GT green, predictions orange)

## Repro

```
python -m src.detection.run_benchmark --limit 50
```
