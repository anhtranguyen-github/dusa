# Buổi 2 — Text Recognition (OCR) + Repo-wide uv migration

**Date:** 2026-06-14
**Session:** Buổi 2 (Thứ 5, 04/06) — Text Recognition
**Curriculum source:** `docs/MasterClass AI for Document Understanding  - Nội dung chương trình.pdf` p.1
**Author:** tra01 (with Claude)

## 1. Goal

Ship Buổi 2 end-to-end while migrating the whole project to a per-notebook
`uv pip install --system` dependency pattern, so every session's notebook is
self-bootstrapping in both local and Colab.

Two deliverables in one commit:

1. **Buổi 2 OCR session** — fine-tune PARSeq on SROIE line crops, compare against
   CRNN + TransformerOCR baselines, emit the OCR JSON that S3 & S4 consume.
2. **Repo-wide notebook pattern** — every notebook (S1–S10) starts with the same
   three-cell uv environment block and owns its own `requirements/buoi-N.txt`.
   S3–S10 ship as stubs in this commit; only S1 and S2 are functional.

## 2. Non-goals

- No `pyproject.toml` migration. The per-notebook requirements files are the
  source of truth.
- No MixNet re-training in this session. S1 currently produces only
  `mixnet_sroie_smoketest.pth`; Buổi 2 uses **GT polygon crops** as the
  recognition source. The realistic MixNet→PARSeq coupling is evaluated in
  the S9 capstone, not here.
- No VLM, no Qwen, no Triton. Those are later sessions; their notebook stubs
  only declare deps, no logic.

## 3. The repo-wide uv pattern

### 3.1 Three cells, identical in every notebook

**Cell 1 (markdown)** — session header.

**Cell 2 (code)** — writes the session's requirements file:

```python
%%writefile requirements/buoi-2.txt
torch>=2.2
torchvision
transformers>=4.40
... (session-specific list)
```

**Cell 3 (code)** — installs with uv, falling back to pip-install-uv first if
absent:

```python
import shutil, subprocess, sys
if shutil.which("uv") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
subprocess.check_call(["uv", "pip", "install", "--system", "-q", "-r", "requirements/buoi-2.txt"])
```

Rationale for `--system`:
- Colab has no usable venv at kernel start; `--system` installs into the kernel's
  active interpreter so subsequent `import` cells just work.
- Local users run inside their own `.venv` — `--system` then means "install into
  *this* interpreter," which is what we want.

### 3.2 Why per-notebook, not a shared requirements

- Each notebook is self-contained for Colab (no `git clone` of stale repo state
  required to install deps; Colab can paste the notebook and go).
- S6's ONNX/Triton stack doesn't force S2 students to install vLLM.
- The `%%writefile` cell is the source of truth at runtime; the
  `requirements/buoi-N.txt` checked into git is the *materialized* version so
  reviewers and `make_colab_notebook.py` can read it without executing the
  notebook.
- Duplication of common deps (numpy, torch) across files is accepted — it's the
  cost of removing cross-session coupling.

### 3.3 Colab variant

`scripts/make_colab_notebook.py` prepends a bootstrap to the local notebook:

```python
!pip install -q uv
!git clone -q https://github.com/anhtranguyen-github/dusa.git
%cd dusa
```

Then copies the local notebook's cells unchanged. The Colab badge link points
to `notebooks/colab/<file>.ipynb`.

### 3.4 What gets deleted

- `requirements.txt` (root) — replaced by per-session files.
- The `README.md` "Quick start" section is rewritten to point users at the
  notebook for their session.

## 4. Buổi 2 — OCR session content

### 4.1 Pipeline this notebook builds

```
SROIE GT polygons ─► line crops ─► train PARSeq
                                 ├─► eval CRNN baseline    \
                                 ├─► eval TransformerOCR    ├─► CER/WER table
                                 └─► eval PARSeq fine-tuned/
                                                 │
                                                 ▼
                                       OCR JSON for all test images
                                       (feeds S3 LayoutLMv3 + S4 Qwen)
```

### 4.2 Models (3-way comparison)

| Model | Mode | Source | Notes |
|---|---|---|---|
| CRNN | eval-only baseline | best-effort HF mirror; fallback = tiny CRNN trained from scratch 2–3 epochs | Documents the classic CTC baseline. If no clean weights load, train tiny from scratch (~5 min on RTX 4060). |
| TransformerOCR | zero-shot eval | `microsoft/trocr-base-printed` | Cross-attention encoder-decoder; no fine-tune. |
| PARSeq | fine-tune | `baudm/parseq-tiny` via `torch.hub` | Existing `src/recognition/train_parseq.py` is wired. |

### 4.3 Crop source

Use existing `src/data/prep_recognition.py` — axis-aligned crops from SROIE GT
line polygons. The docstring already flags how to swap in MixNet predictions
later; that swap belongs to S9, not here.

### 4.4 Artifacts produced

| Artifact | Path |
|---|---|
| Local notebook | `notebooks/02_ocr_finetune_parseq.ipynb` |
| Colab notebook | `notebooks/colab/02_ocr_finetune_parseq.ipynb` |
| Crops + labels | `data/processed/recognition/{train,test}/images/*.jpg`, `labels.txt` |
| PARSeq checkpoint | `checkpoints/recognition/parseq_sroie_finetuned.ckpt` |
| Training history | `checkpoints/recognition/parseq_metrics_history.jsonl` |
| Comparison table | `reports/recognition/cer_wer_comparison.json`, `.md` |
| OCR predictions for S3/S4 | `data/processed/recognition/ocr_predictions_test.json` |
| Visualizations (PNG, all committed) | `reports/recognition/figures/*.png` |
| Session requirements | `requirements/buoi-2.txt` |

### 4.5 Saved visualizations

All six render to PNGs under `reports/recognition/figures/`. The notebook also
displays them inline.

| # | File | What it shows |
|---|---|---|
| 1 | `crop_grid.png` | 5×4 random training crops with GT text |
| 2 | `label_length_hist.png` | Label length distribution (train vs test) |
| 3 | `char_freq.png` | Top-50 character frequency |
| 4 | `training_curves.png` | Loss / CER / WER per epoch (from `parseq_metrics_history.jsonl`) |
| 5 | `prediction_grid.png` | 6×4 grid per model (CRNN / TrOCR / PARSeq) — crop + GT (green) + pred (red if wrong) |
| 6 | `error_examples.png` | Top-10 worst CER crops with GT vs pred + edit distance |

### 4.6 OCR JSON schema (input to S3/S4)

`data/processed/recognition/ocr_predictions_test.json`:

```json
{
  "X51005200931": [
    {"bbox": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]], "text": "TESCO STORES", "conf": null},
    ...
  ],
  ...
}
```

`conf` is `null` because PARSeq doesn't expose per-line confidence in the
torch.hub interface; S3 doesn't use confidence, so the field is reserved.
`bbox` is the GT polygon used to make the crop — keeping it here means S3 can
align text to layout without re-running detection.

### 4.7 Notebook structure (local; ~16 cells)

1. Markdown header — "Buổi 2 — Text Recognition (OCR)"
2. `%%writefile requirements/buoi-2.txt` (uv pattern, cell 2)
3. uv install (uv pattern, cell 3)
4. Imports + paths (auto-detect `IN_COLAB`)
5. Run `python -m src.data.prep_recognition` if `data/processed/recognition/` is empty
6. Dataset stats + crop grid + label length hist + char freq (3 figures)
7. Theory recap (markdown) — CTC vs cross-attention vs permutation LM, 3-column table
8. Evaluator helper class (inline, ~20 lines wrapping `src.recognition.eval`)
9. CRNN baseline eval
10. TransformerOCR eval
11. PARSeq fine-tune call + training_curves.png
12. PARSeq eval
13. Comparison table → save JSON, MD, and bar-chart PNG
14. Prediction grid + error examples (figs 5–6)
15. OCR JSON export over full SROIE test set
16. Wrap-up markdown — error patterns, what S3 inherits

### 4.8 New / modified source files

**New:**

- `src/recognition/eval.py` — `CRNNAdapter`, `TrOCRAdapter`, `PARSeqAdapter`
  (uniform `predict(crops) -> list[str]`), `RecogEvaluator` (CER/WER aggregation,
  worst-N retrieval).
- `src/recognition/export_predictions.py` — runs PARSeq over SROIE test,
  emits `ocr_predictions_test.json`. Callable as
  `python -m src.recognition.export_predictions`.
- `src/recognition/viz.py` — six plotting functions, pure matplotlib, save-only,
  one function per figure.

**Modified:**

- `src/recognition/train_parseq.py` — emit per-epoch
  `parseq_metrics_history.jsonl` so the notebook can plot curves without
  re-running training.

**Unchanged:**

- `src/data/prep_recognition.py`
- `src/recognition/dataset.py`
- `configs/recognition/parseq_sroie.yaml`

### 4.9 CRNN fallback policy

Try in order, commit whichever loads cleanly:

1. A working pretrained CRNN from HuggingFace (e.g., search for
   `*crnn*` on receipts/printed text).
2. PaddleOCR's `en_PP-OCRv3_rec` wrapped through `paddleocr`'s Python API.
3. Train a tiny CRNN (`src/recognition/crnn_tiny.py`, ~80 lines) from scratch on
   the SROIE crops for 2–3 epochs.

Whichever path wins is documented in the comparison report.

## 5. All-notebooks touch list

| Notebook | Action this commit | Functional? |
|---|---|---|
| `01_detection_benchmark.ipynb` | Replace bootstrap cells with 3-cell uv pattern | Yes (existing logic) |
| `02_ocr_finetune_parseq.ipynb` | Full build per §4 | Yes |
| `03_kie_layoutlmv3_finetune.ipynb` | Stub: 3-cell uv pattern + agenda markdown | No |
| `04_qwen_kie_zeroshot_and_qlora.ipynb` | Stub | No |
| `05_vlm_zeroshot_demo.ipynb` | Stub | No |
| `06_onnx_triton_export.ipynb` | Stub | No |
| `07_triton_ensemble_vllm.ipynb` | Stub | No |
| `08_fastapi_dual_backend.ipynb` | Stub | No |
| `09_capstone_e2e.ipynb` | Stub | No |
| `10_demo_final.ipynb` | Stub | No |
| `notebooks/colab/01_…ipynb` | Regenerated by updated script | Yes |
| `notebooks/colab/02_…ipynb` | Generated for the first time | Yes |
| `notebooks/colab/03–10` | Not generated this commit | — |

Stub = 3 cells per the uv pattern, plus a markdown agenda derived from the PDF.
No imports, no model code. Each stub's `requirements/buoi-N.txt` carries that
session's dep list per §3.3.

## 6. `scripts/make_colab_notebook.py` refactor

Currently bakes S1's Colab variant via hard-coded constants. Refactor to:

- Accept a `--session N` arg.
- Read `notebooks/0N_<name>.ipynb`, prepend a uniform Colab bootstrap, write to
  `notebooks/colab/0N_<name>.ipynb`.
- Drop the per-session VRAM/batch tweaks for now — Colab T4 has 15 GB and we're
  not exceeding that in S2. (S1's existing batch tweak stays as a session-1
  special case.)
- Run S1 + S2 in this commit:
  ```
  python scripts/make_colab_notebook.py --session 1
  python scripts/make_colab_notebook.py --session 2
  ```

## 7. Verification

Before claiming done:

1. `uv pip install --system -r requirements/buoi-2.txt` succeeds on a clean shell.
2. `python -m src.data.prep_recognition` writes non-empty `labels.txt` for both
   splits.
3. `python -m src.recognition.train_parseq --config configs/recognition/parseq_sroie.yaml --max_steps 20`
   runs end-to-end on CPU without import errors (smoke test).
4. Full PARSeq fine-tune on the RTX 4060 achieves **CER < 0.10** on SROIE test
   (sanity gate — receipts are clean printed text).
5. `data/processed/recognition/ocr_predictions_test.json` has one entry per
   SROIE test image, schema matches §4.6.
6. All six PNGs in `reports/recognition/figures/` exist and are non-empty.
7. The CER/WER comparison report shows three rows: CRNN, TrOCR, PARSeq.
8. Stub notebooks S3–S10 each open in Jupyter without error (nbformat valid,
   3-cell uv pattern present).
9. `notebooks/colab/02_…ipynb` is generated and its bootstrap cell uses uv.

## 8. Risks / open items

| Risk | Mitigation |
|---|---|
| `baudm/parseq` torch.hub load fails | Fallback documented in `train_parseq.py`: `uv pip install --system 'git+https://github.com/baudm/parseq.git'`. |
| CRNN HF pretrained weights are noisy / nonexistent | Tiny-from-scratch fallback (§4.9). |
| `microsoft/trocr-base-printed` is large for Colab cold start | Acceptable — T4 has 15 GB; we only need inference. |
| Colab `!uv pip install --system` race with already-installed torch | uv resolves to existing torch on Colab; we don't pin torch version. |
| Stub notebooks' `requirements/buoi-N.txt` lists may bit-rot before that session | Accepted — they're best-guess from the curriculum; updated when each session is built out. |

## 9. Out of scope (explicit)

- MixNet re-training in S1 (separate work).
- Donut / GOT-OCR variants (S5 territory).
- Any KIE work (S3+).
- Vietnamese receipts (VN Receipts dataset is in `data/` but not used here).
