# dusa — Document Understanding System (SROIE)

End-to-end Document AI pipeline built across the **MasterClass: AI for Document Understanding** (10 sessions, 01.06.2026 → 02.07.2026, mentor: Ths. Nguyễn Việt Hoài).

Dataset: **SROIE** (~1,000 receipt images, 4 KIE fields: `company`, `date`, `address`, `total`).

## Pipeline

```
Image ─► Layout (DocLayout-YOLO) ─► Detection (MixNet) ─► Recognition (PARSeq)
                                                                │
                                                                ▼
                                          KIE backend ──► JSON { company, date, address, total }
                                          ├── LayoutLMv3 (Triton, ONNX)
                                          └── Qwen2.5-3B + LoRA (vLLM)
                                                                │
                                                                ▼
                                                  FastAPI: POST /kie?backend=…
```

## Project layout

```
dusa/
├── docs/                       Course materials (PDF, poster)
├── src/                        Library code
│   ├── layout/                 DocLayout-YOLO (S1)
│   ├── detection/              DB-Net, MixNet (S1)
│   ├── recognition/            CRNN, TransformerOCR, PARSeq (S2)
│   ├── kie/
│   │   ├── layoutlm/           LayoutLMv3 fine-tune (S3) — KIE backend #1
│   │   └── qwen/               Qwen2.5-3B zero-shot + QLoRA (S4) — KIE backend #2
│   ├── vlm/                    GLM-OCR / Nanonets-OCR demos (S5)
│   ├── pipeline/               E2E orchestration
│   ├── serving/
│   │   ├── onnx_export/        PyTorch → ONNX (S6)
│   │   └── clients/            Triton gRPC + vLLM HTTP clients
│   ├── api/                    FastAPI app
│   │   ├── routers/            /kie, /health, /models/status
│   │   ├── middleware/         CORS, logging, request tracing
│   │   └── schemas.py          Pydantic models
│   ├── data/                   SROIE loaders, BIO alignment, instruction-dataset builder
│   ├── eval/                   CER / WER / F1 / Exact Match
│   └── utils/
├── notebooks/                  One per session
├── configs/                    YAML/JSON training & serving configs
├── triton/model_repository/    parseq/, layoutlmv3/, ensemble_kie/ (config.pbtxt)
├── docker/                     Dockerfiles + docker-compose.yml
├── data/                       SROIE (gitignored)
├── checkpoints/                Weights, LoRA adapters, ONNX (gitignored)
├── scripts/                    Download SROIE, prepare data, export ONNX, benchmarks
├── tests/                      Unit / integration tests
└── reports/                    F1 per field, CER/WER, latency, throughput
```

## Session → artifact map

| #  | Date  | Notebook                                 | Key outputs                                            |
|----|-------|------------------------------------------|--------------------------------------------------------|
| 1  | 01/06 | `01_detection_benchmark.ipynb`           | `mixnet_sroie_finetuned.pth`, DB-Net vs MixNet F1     |
| 2  | 04/06 | `02_ocr_finetune_parseq.ipynb`           | `parseq_sroie_finetuned.ckpt`, CER/WER table          |
| 3  | 08/06 | `03_kie_layoutlmv3_finetune.ipynb`       | `layoutlmv3_sroie_kie.bin` — KIE backend #1           |
| 4  | 11/06 | `04_qwen_kie_zeroshot_and_qlora.ipynb`   | Zero-shot F1; **if < 0.85** → `qwen25_3b_sroie_lora/` |
| 5  | 15/06 | `05_vlm_zeroshot_demo.ipynb`             | GLM-OCR / Nanonets-OCR demo (theory + no fine-tune)   |
| 6  | 18/06 | `06_onnx_triton_export.ipynb`            | `parseq.onnx`, `layoutlmv3.onnx` on Triton            |
| 7  | 22/06 | `07_triton_ensemble_vllm.ipynb`          | Triton Ensemble DAG + vLLM serving Qwen2.5-3B         |
| 8  | 25/06 | `08_fastapi_dual_backend.ipynb`          | `POST /kie?backend={layoutlmv3\|qwen}`                |
| 9  | 29/06 | `09_capstone_e2e.ipynb`                  | Full pipeline; F1 per field, E2E latency, throughput  |
| 10 | 02/07 | `10_demo_final.ipynb`                    | Demo slides, roadmap                                  |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch SROIE (652 receipts → 522 train / 130 test, deterministic seed=42)
python scripts/download_sroie.py

# Run FastAPI (after S8)
uvicorn src.api.main:app --reload
```

## Datasets

The architecture in `docs/` lists 5 input sources: **SROIE / VN Receipts / FUNSD / CORD / Custom**. The first four are downloadable from HuggingFace; "Custom" is per-team data. SROIE remains the spine dataset for fine-tuning across all 10 sessions — the others are included so the pipeline can be evaluated against forms (FUNSD), Indonesian restaurant receipts (CORD), and Vietnamese tax receipts (MC-OCR).

| Dataset      | HF source                                      | Splits (train/val/test) | Annotation                                                          |
|--------------|------------------------------------------------|------------------------:|---------------------------------------------------------------------|
| SROIE        | `arvindrajan92/sroie_document_understanding`   |        522 / – / 130    | Polygon bbox + line text + KIE label (company/date/address/total)   |
| FUNSD        | `nielsr/funsd`                                 |        149 / – / 50     | Word bbox + text + BIO (HEADER / QUESTION / ANSWER)                 |
| CORD         | `naver-clova-ix/cord-v2`                       |        800 / 100 / 100  | Donut-style nested JSON (menu / sub_total / total)                  |
| VN Receipts  | `DThai/mcocr-test` (MC-OCR 2021)               |       1199 / 289 / 296  | Donut-style JSON (Số hoá đơn / Ngày / Tổng / restaurant fields)     |
| Custom       | per-team                                       |              –          | per-team format                                                     |

Fetch with:

```bash
python scripts/download_sroie.py
python scripts/download_funsd.py
python scripts/download_cord.py
python scripts/download_vn_receipts.py
```

Each script materializes the same layout (so a single loader can switch datasets):

```
data/<name>/
├── images/{train,val,test}/<id>.{jpg,png}
├── annotations/{train,val,test}/<id>.json
└── splits/{train,val,test}.txt
```

Annotation format per file:
- **SROIE & FUNSD**: list of `{box, label, text}` (per line / per word).
- **CORD & VN Receipts**: parsed `gt_parse` dict from the Donut ground-truth (hierarchical, schema varies per receipt).

## Conditional fine-tune rule (S4)

Qwen2.5-3B is only fine-tuned when zero-shot **Macro F1 < 0.85** on SROIE test. Otherwise log the zero-shot result and skip fine-tuning.
