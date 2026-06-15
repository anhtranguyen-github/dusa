"""KIE evaluation metrics for Qwen JSON outputs vs SROIE ground truth.

Per-field exact match + F1 (treating each field as a binary classification of
whether the predicted string equals the GT string after normalization).
Also reports a Macro-F1 across the 4 fields — the gate metric in the curriculum.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

FIELDS = ("company", "date", "address", "total")


def _normalize(s: str | None) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[\.,\-\:\(\)\[\]\"']", "", s)
    return s


def _exact(pred: str | None, gold: str | None) -> bool:
    return _normalize(pred) == _normalize(gold) and _normalize(gold) != ""


def _binary_f1(matches: list[bool], gold_present: list[bool], pred_present: list[bool]) -> dict:
    tp = sum(1 for m, g, p in zip(matches, gold_present, pred_present) if m and g and p)
    fp = sum(1 for m, g, p in zip(matches, gold_present, pred_present) if (not m) and p)
    fn = sum(1 for m, g, p in zip(matches, gold_present, pred_present) if (not m) and g)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4), "tp": tp, "fp": fp, "fn": fn}


@dataclass
class KIEEvalResult:
    per_field: dict
    macro_f1: float
    exact_match_rate: float
    n: int

    def to_dict(self) -> dict:
        return {
            "per_field": self.per_field,
            "macro_f1": round(self.macro_f1, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "n": self.n,
        }


def evaluate(predictions: dict[str, dict], targets: dict[str, dict]) -> KIEEvalResult:
    ids = sorted(set(predictions) & set(targets))
    per_field: dict[str, dict] = {}
    for f in FIELDS:
        matches, golds, preds = [], [], []
        for i in ids:
            g = targets[i].get(f)
            p = predictions[i].get(f)
            matches.append(_exact(p, g))
            golds.append(bool((g or "").strip()) if isinstance(g, str) else g is not None)
            preds.append(bool((p or "").strip()) if isinstance(p, str) else p is not None)
        per_field[f] = _binary_f1(matches, golds, preds)
    macro_f1 = sum(per_field[f]["F1"] for f in FIELDS) / len(FIELDS)
    em_rate = sum(
        all(_exact(predictions[i].get(f), targets[i].get(f)) for f in FIELDS) for i in ids
    ) / max(len(ids), 1)
    return KIEEvalResult(per_field=per_field, macro_f1=macro_f1, exact_match_rate=em_rate, n=len(ids))


def parse_json_output(text: str) -> dict:
    """Extract the first {...} block from a model response."""
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
