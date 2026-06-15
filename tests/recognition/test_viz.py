from pathlib import Path

import numpy as np
import pytest

from src.recognition.viz import (
    save_crop_grid,
    save_label_length_hist,
    save_char_freq,
    save_training_curves,
    save_prediction_grid,
    save_error_examples,
)


def _png_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 1_000


@pytest.fixture
def fig_dir(tmp_path):
    d = tmp_path / "figs"
    d.mkdir()
    return d


def _fake_crops(n=8):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (32, 128, 3), dtype=np.uint8) for _ in range(n)]


def test_crop_grid(fig_dir):
    out = fig_dir / "crop_grid.png"
    save_crop_grid(_fake_crops(8), ["t" + str(i) for i in range(8)], out, cols=4)
    assert _png_nonempty(out)


def test_label_length_hist(fig_dir):
    out = fig_dir / "label_length_hist.png"
    save_label_length_hist({"train": [3, 5, 8, 8, 12], "test": [4, 4, 9]}, out)
    assert _png_nonempty(out)


def test_char_freq(fig_dir):
    out = fig_dir / "char_freq.png"
    save_char_freq(["hello world", "TESCO 123"], out, top_k=10)
    assert _png_nonempty(out)


def test_training_curves(fig_dir):
    out = fig_dir / "training_curves.png"
    history = [
        {"epoch": 0, "loss": 2.5, "cer": 0.5, "wer": 0.7},
        {"epoch": 1, "loss": 1.2, "cer": 0.2, "wer": 0.3},
        {"epoch": 2, "loss": 0.6, "cer": 0.05, "wer": 0.1},
    ]
    save_training_curves(history, out)
    assert _png_nonempty(out)


def test_prediction_grid(fig_dir):
    out = fig_dir / "prediction_grid.png"
    crops = _fake_crops(6)
    save_prediction_grid(
        crops=crops,
        gts=["a", "b", "c", "d", "e", "f"],
        preds_by_model={
            "crnn":   ["a", "x", "c", "d", "e", "f"],
            "trocr":  ["a", "b", "c", "z", "e", "f"],
            "parseq": ["a", "b", "c", "d", "e", "f"],
        },
        out_path=out,
        cols=3,
    )
    assert _png_nonempty(out)


def test_error_examples(fig_dir):
    out = fig_dir / "error_examples.png"
    crops = _fake_crops(4)
    save_error_examples(
        crops=crops,
        worst=[
            {"pred": "x", "gt": "abc", "cer": 1.0},
            {"pred": "te", "gt": "tesc", "cer": 0.5},
            {"pred": "totl", "gt": "total", "cer": 0.2},
            {"pred": "data", "gt": "date", "cer": 0.5},
        ],
        out_path=out,
    )
    assert _png_nonempty(out)
