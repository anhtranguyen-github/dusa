"""One-shot writer for the Buổi 2 notebook. Re-run if cells need updating."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("notebooks/02_ocr_finetune_parseq.ipynb")


def md(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


CELLS = [
    md("s2-header", """\
# Buổi 2 — Text Recognition (OCR)

Fine-tune PARSeq on SROIE line crops, compare against CRNN baseline and TrOCR
(zero-shot), and export OCR JSON for S3 LayoutLMv3 / S4 Qwen.

Curriculum spec: `docs/MasterClass …pdf` (Buổi 2, 04/06).
"""),
    code("s2-req", """\
%%writefile requirements/buoi-2.txt
# requirements/buoi-2.txt — Buổi 2: Text Recognition (OCR)
torch>=2.2
torchvision
transformers>=4.40
datasets
editdistance
opencv-python
pillow
numpy
pyyaml
matplotlib
tqdm
timm                  # PARSeq vit_small backbone
ftfy                  # transformers tokenizer helper
sentencepiece         # trocr tokenizer
"""),
    code("s2-uv", """\
import shutil, subprocess, sys
if shutil.which("uv") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
subprocess.check_call(["uv", "pip", "install", "--system", "-q",
                       "-r", "requirements/buoi-2.txt"])
"""),
    code("s2-imports", """\
import json, os, sys, random, subprocess
from pathlib import Path

import numpy as np

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

PROJECT_ROOT = Path("/content/dusa") if IN_COLAB else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

random.seed(42); np.random.seed(42)
print("cwd:", PROJECT_ROOT, "  colab:", IN_COLAB)
"""),
    md("s2-prep-header", "## 1. Data prep — crop SROIE lines for recognition\n"),
    code("s2-prep", """\
# Crop SROIE GT lines if not done yet.
if not (PROJECT_ROOT / "data/processed/recognition/train/labels.txt").exists():
    subprocess.check_call([sys.executable, "-m", "src.data.prep_recognition"])
print("train labels exist:",
      (PROJECT_ROOT / "data/processed/recognition/train/labels.txt").exists())
print("test  labels exist:",
      (PROJECT_ROOT / "data/processed/recognition/test/labels.txt").exists())
"""),
    md("s2-viz-header", "## 2. Dataset stats + visualizations\n"),
    code("s2-viz", """\
import cv2
from src.recognition.viz import save_crop_grid, save_label_length_hist, save_char_freq

FIG = PROJECT_ROOT / "reports/recognition/figures"
FIG.mkdir(parents=True, exist_ok=True)

def _load_labels(split):
    rows = []
    root = PROJECT_ROOT / f"data/processed/recognition/{split}"
    for line in (root / "labels.txt").read_text().splitlines():
        if not line.strip():
            continue
        rel, text = line.split("\\t", 1)
        rows.append((root / rel, text))
    return rows

train_rows = _load_labels("train")
test_rows  = _load_labels("test")
print(f"train crops: {len(train_rows)}  test crops: {len(test_rows)}")

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(train_rows), size=min(20, len(train_rows)), replace=False)
sample = [train_rows[i] for i in sample_idx]
imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p, _ in sample]
texts = [t for _, t in sample]
save_crop_grid(imgs, texts, FIG / "crop_grid.png", cols=4)

save_label_length_hist(
    {"train": [len(t) for _, t in train_rows], "test": [len(t) for _, t in test_rows]},
    FIG / "label_length_hist.png",
)
save_char_freq([t for _, t in train_rows], FIG / "char_freq.png", top_k=50)

from IPython.display import Image, display
for name in ("crop_grid", "label_length_hist", "char_freq"):
    display(Image(str(FIG / f"{name}.png")))
"""),
    md("s2-theory", """\
## 3. Theory recap

| Model | Decoder | Trains on | Strengths | Weaknesses |
|---|---|---|---|---|
| **CRNN** (CNN+BiLSTM+CTC) | CTC greedy/beam | Word/line images + transcripts | Fast, simple, robust to length variance | No language model, weak on cursive |
| **TransformerOCR** (TrOCR) | Cross-attention encoder-decoder | Image patches → tokens (BPE) | Strong implicit LM, handles long lines | Slow autoregressive decode |
| **PARSeq** | Permutation LM (multi-order AR) | Image patches → chars | Robust to occlusion, no explicit LM, beats TrOCR on STR benchmarks | Char-level only, not great for rare subwords |

We fine-tune **PARSeq** on SROIE GT crops; CRNN and TrOCR are evaluated zero-shot as baselines.
"""),
    md("s2-baselines-header", "## 4. Baselines — CRNN + TrOCR zero-shot\n"),
    code("s2-baselines", """\
from src.recognition.adapters import TrOCRAdapter, CRNNAdapter
from src.recognition.evaluator import RecogEvaluator

EVAL_N = 200 if not IN_COLAB else 130
test_subset = test_rows[:EVAL_N]
test_imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p, _ in test_subset]
test_gts = [t for _, t in test_subset]

def _eval(adapter):
    preds = adapter.predict(test_imgs)
    ev = RecogEvaluator(); ev.add(preds, test_gts)
    return ev, preds

print("CRNN (tiny, untrained) — sanity baseline")
crnn_ev, crnn_preds = _eval(CRNNAdapter())
print(f"  CER={crnn_ev.compute()['cer']:.4f}  WER={crnn_ev.compute()['wer']:.4f}")

print("TrOCR (microsoft/trocr-base-printed) — zero-shot")
trocr_ev, trocr_preds = _eval(TrOCRAdapter())
print(f"  CER={trocr_ev.compute()['cer']:.4f}  WER={trocr_ev.compute()['wer']:.4f}")
"""),
    md("s2-finetune-header", "## 5. Fine-tune PARSeq\n"),
    code("s2-finetune", """\
CKPT = PROJECT_ROOT / "checkpoints/recognition/parseq_sroie_finetuned.ckpt"
if not CKPT.exists():
    subprocess.check_call([
        sys.executable, "-m", "src.recognition.train_parseq",
        "--config", "configs/recognition/parseq_sroie.yaml",
    ])
print("checkpoint exists:", CKPT.exists())
"""),
    code("s2-curves", """\
from src.recognition.viz import save_training_curves
HISTORY = PROJECT_ROOT / "checkpoints/recognition/parseq_metrics_history.jsonl"
if HISTORY.exists():
    history = [json.loads(line) for line in HISTORY.read_text().splitlines() if line.strip()]
    save_training_curves(history, FIG / "training_curves.png")
    from IPython.display import Image, display
    display(Image(str(FIG / "training_curves.png")))
else:
    print("no history — was train_parseq skipped?")
"""),
    md("s2-parseq-eval-header", "## 6. Evaluate PARSeq fine-tuned\n"),
    code("s2-parseq-eval", """\
from src.recognition.adapters import PARSeqAdapter
parseq = PARSeqAdapter(checkpoint_path=str(CKPT) if CKPT.exists() else None)
parseq_ev, parseq_preds = _eval(parseq)
print(f"  CER={parseq_ev.compute()['cer']:.4f}  WER={parseq_ev.compute()['wer']:.4f}")
"""),
    md("s2-cmp-header", "## 7. Comparison table\n"),
    code("s2-cmp", """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = {
    "crnn":   crnn_ev.compute(),
    "trocr":  trocr_ev.compute(),
    "parseq": parseq_ev.compute(),
}

REP = PROJECT_ROOT / "reports/recognition"
REP.mkdir(parents=True, exist_ok=True)
(REP / "cer_wer_comparison.json").write_text(json.dumps(results, indent=2))

md_lines = ["| model | CER | WER | n |", "|---|---|---|---|"]
for m, r in results.items():
    md_lines.append(f"| {m} | {r['cer']:.4f} | {r['wer']:.4f} | {r['n']} |")
(REP / "cer_wer_comparison.md").write_text("\\n".join(md_lines) + "\\n")
print("\\n".join(md_lines))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
names = list(results.keys())
ax[0].bar(names, [results[m]["cer"] for m in names]); ax[0].set_title("CER (lower is better)")
ax[1].bar(names, [results[m]["wer"] for m in names]); ax[1].set_title("WER (lower is better)")
fig.tight_layout(); fig.savefig(FIG / "comparison_bars.png", dpi=110); plt.close(fig)
from IPython.display import Image, display
display(Image(str(FIG / "comparison_bars.png")))
"""),
    md("s2-pred-viz-header", "## 8. Prediction grids + worst errors\n"),
    code("s2-pred-viz", """\
from src.recognition.viz import save_prediction_grid, save_error_examples

show_n = 12
crops_show = test_imgs[:show_n]
gts_show = test_gts[:show_n]
preds = {"crnn": crnn_preds[:show_n], "trocr": trocr_preds[:show_n], "parseq": parseq_preds[:show_n]}
save_prediction_grid(crops_show, gts_show, preds, FIG / "prediction_grid.png", cols=3)

worst = parseq_ev.worst_n(10)
worst_imgs = []
for w in worst:
    if w["gt"] in test_gts:
        idx = test_gts.index(w["gt"])
        worst_imgs.append(test_imgs[idx])
save_error_examples(worst_imgs, worst[:len(worst_imgs)], FIG / "error_examples.png")

from IPython.display import Image, display
for name in ("prediction_grid", "error_examples"):
    display(Image(str(FIG / f"{name}.png")))
"""),
    md("s2-export-header", "## 9. Export OCR JSON for S3 / S4\n"),
    code("s2-export", """\
OUT = PROJECT_ROOT / "data/processed/recognition/ocr_predictions_test.json"
cmd = [sys.executable, "-m", "src.recognition.export_predictions",
       "--out", str(OUT.relative_to(PROJECT_ROOT))]
if CKPT.exists():
    cmd += ["--checkpoint", str(CKPT)]
subprocess.check_call(cmd)
print("exists:", OUT.exists(), "  size:", OUT.stat().st_size, "B")
"""),
    md("s2-wrap", """\
## 10. Wrap-up

- PARSeq fine-tuned on SROIE GT crops produces the OCR JSON consumed by **S3 LayoutLMv3** and **S4 Qwen**.
- The realistic detection→recognition coupling (MixNet boxes feeding PARSeq) is evaluated end-to-end in the **S9 capstone**, not here.
- Error patterns to flag in S3: long lines truncated at 32 chars, special symbols (`$`, `*`, `:`), and uppercase/lowercase confusions.
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
