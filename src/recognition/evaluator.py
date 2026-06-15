"""CER/WER aggregator + worst-sample retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field

import editdistance


@dataclass
class _Sample:
    pred: str
    gt: str
    cer: float
    wer: float


def _cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return editdistance.eval(pred, gt) / len(gt)


def _wer(pred: str, gt: str) -> float:
    gt_words = gt.split()
    if not gt_words:
        return 0.0 if not pred.split() else 1.0
    return editdistance.eval(pred.split(), gt_words) / len(gt_words)


@dataclass
class RecogEvaluator:
    samples: list[_Sample] = field(default_factory=list)

    def add(self, preds: list[str], gts: list[str]) -> None:
        if len(preds) != len(gts):
            raise ValueError(f"preds/gts mismatch: {len(preds)} vs {len(gts)}")
        for p, g in zip(preds, gts):
            self.samples.append(_Sample(p, g, _cer(p, g), _wer(p, g)))

    def compute(self) -> dict:
        if not self.samples:
            return {"cer": 0.0, "wer": 0.0, "n": 0}
        return {
            "cer": sum(s.cer for s in self.samples) / len(self.samples),
            "wer": sum(s.wer for s in self.samples) / len(self.samples),
            "n": len(self.samples),
        }

    def worst_n(self, n: int) -> list[dict]:
        ordered = sorted(self.samples, key=lambda s: s.cer, reverse=True)
        return [{"pred": s.pred, "gt": s.gt, "cer": s.cer, "wer": s.wer}
                for s in ordered[:n]]
