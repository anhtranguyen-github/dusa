"""Saved-PNG visualizations for the Buổi 2 OCR notebook.

All functions write to disk and return None. Use a non-interactive matplotlib
backend so tests run headless.
"""
from __future__ import annotations

import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _new_fig(w: float = 12, h: float = 6):
    return plt.figure(figsize=(w, h))


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_crop_grid(crops: list[np.ndarray], texts: list[str], out_path: Path,
                   cols: int = 4) -> None:
    n = len(crops)
    rows = (n + cols - 1) // cols
    fig = _new_fig(w=cols * 3, h=rows * 1.6)
    for i, (img, txt) in enumerate(zip(crops, texts)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(txt[:30], fontsize=8)
        ax.axis("off")
    _save(fig, out_path)


def save_label_length_hist(lengths_by_split: dict[str, list[int]], out_path: Path) -> None:
    fig = _new_fig(8, 4)
    ax = fig.add_subplot(1, 1, 1)
    bins = range(0, max(max(v, default=0) for v in lengths_by_split.values()) + 5, 2)
    for name, lens in lengths_by_split.items():
        ax.hist(lens, bins=list(bins), alpha=0.5, label=name)
    ax.set_xlabel("label length (chars)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Label length distribution")
    _save(fig, out_path)


def save_char_freq(texts: list[str], out_path: Path, top_k: int = 50) -> None:
    counter: collections.Counter[str] = collections.Counter()
    for t in texts:
        counter.update(t)
    items = counter.most_common(top_k)
    fig = _new_fig(max(8, len(items) * 0.25), 4)
    ax = fig.add_subplot(1, 1, 1)
    labels = [f"'{c}'" if c != " " else "' '" for c, _ in items]
    ax.bar(range(len(items)), [n for _, n in items])
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=70, fontsize=8)
    ax.set_title(f"Top-{len(items)} character frequency")
    _save(fig, out_path)


def save_training_curves(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig = _new_fig(12, 4)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, [h["loss"] for h in history], marker="o")
    ax1.set_title("loss"); ax1.set_xlabel("epoch")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, [h["cer"] for h in history], marker="o", color="tab:orange")
    ax2.set_title("CER (test)"); ax2.set_xlabel("epoch")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(epochs, [h["wer"] for h in history], marker="o", color="tab:green")
    ax3.set_title("WER (test)"); ax3.set_xlabel("epoch")
    _save(fig, out_path)


def save_prediction_grid(crops: list[np.ndarray], gts: list[str],
                          preds_by_model: dict[str, list[str]],
                          out_path: Path, cols: int = 3) -> None:
    n = len(crops)
    rows_per_model = (n + cols - 1) // cols
    models = list(preds_by_model.keys())
    fig = _new_fig(w=cols * 3, h=rows_per_model * 1.8 * len(models))
    plot_idx = 1
    for m in models:
        for i in range(n):
            ax = fig.add_subplot(rows_per_model * len(models), cols, plot_idx)
            ax.imshow(crops[i])
            pred = preds_by_model[m][i]
            ok = pred == gts[i]
            color = "black" if ok else "red"
            ax.set_title(f"[{m}] gt='{gts[i][:18]}'\npred='{pred[:18]}'",
                         fontsize=7, color=color)
            ax.axis("off")
            plot_idx += 1
    _save(fig, out_path)


def save_error_examples(crops: list[np.ndarray], worst: list[dict], out_path: Path) -> None:
    n = min(len(crops), len(worst))
    cols = 2
    rows = (n + cols - 1) // cols
    fig = _new_fig(w=cols * 4, h=rows * 2)
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(crops[i])
        w = worst[i]
        ax.set_title(f"CER={w['cer']:.2f}\ngt='{w['gt'][:24]}'\npred='{w['pred'][:24]}'",
                     fontsize=8, color="red")
        ax.axis("off")
    _save(fig, out_path)
