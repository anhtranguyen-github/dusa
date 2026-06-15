"""Verify the per-epoch JSONL writer used by train_parseq."""
import json
from pathlib import Path

from src.recognition.train_parseq import append_history


def test_append_history_appends_one_line_per_call(tmp_path: Path):
    p = tmp_path / "history.jsonl"
    append_history(p, {"epoch": 0, "loss": 2.0, "cer": 0.5, "wer": 0.7})
    append_history(p, {"epoch": 1, "loss": 1.0, "cer": 0.2, "wer": 0.3})
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["epoch"] == 0
    assert json.loads(lines[1])["loss"] == 1.0
