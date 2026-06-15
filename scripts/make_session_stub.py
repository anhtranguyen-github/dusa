"""Build a stub notebook for a single session.

Produces a 4-cell notebook:
  1. Markdown title (from SESSION_TITLES)
  2. %%writefile requirements/buoi-N.txt
  3. uv pip install --system -r requirements/buoi-N.txt
  4. Markdown agenda (caller-supplied, derived from the curriculum PDF)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.make_colab_notebook import SESSION_TITLES, SESSION_FILENAMES


def _mkmd(cid: str, src: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


def _mkcode(cid: str, src: str) -> dict:
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


UV_INSTALL_CELL_SRC = """\
import shutil, subprocess, sys
if shutil.which("uv") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
subprocess.check_call(["uv", "pip", "install", "--system", "-q",
                       "-r", "requirements/buoi-{n}.txt"])
"""


def build_stub(session: int, requirements_lines: list[str], agenda_md: str,
               out_path: Path) -> None:
    title = SESSION_TITLES[session]
    title_md = f"# {title}\n\nDeliverable spec — see `docs/MasterClass …pdf`.\n"

    writefile_src = (f"%%writefile requirements/buoi-{session}.txt\n"
                     + "\n".join(requirements_lines) + "\n")
    install_src = UV_INSTALL_CELL_SRC.format(n=session)

    nb = {
        "cells": [
            _mkmd(f"s{session}-header", title_md),
            _mkcode(f"s{session}-req", writefile_src),
            _mkcode(f"s{session}-uv", install_src),
            _mkmd(f"s{session}-agenda", "## Agenda\n\n" + agenda_md),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1) + "\n")


SESSION_DEPS: dict[int, list[str]] = {
    1: ["torch>=2.2", "torchvision", "transformers>=4.40", "ultralytics",
        "shapely", "pyclipper", "python-doctr[torch]", "opencv-python",
        "pillow", "numpy", "pyyaml", "matplotlib", "tqdm", "timm",
        "editdistance"],
    2: ["torch>=2.2", "torchvision", "transformers>=4.40", "datasets",
        "editdistance", "opencv-python", "pillow", "numpy", "pyyaml",
        "matplotlib", "tqdm", "timm", "ftfy", "sentencepiece"],
    3: ["torch>=2.2", "torchvision", "transformers>=4.40", "seqeval",
        "datasets", "opencv-python", "pillow", "numpy", "pyyaml",
        "matplotlib", "tqdm"],
    4: ["torch>=2.2", "transformers>=4.40", "datasets", "peft", "trl",
        "accelerate", "bitsandbytes", "sentencepiece", "matplotlib", "tqdm"],
    5: ["torch>=2.2", "transformers>=4.40", "pillow", "matplotlib",
        "opencv-python", "tqdm", "accelerate", "sentencepiece"],
    6: ["torch>=2.2", "transformers>=4.40", "onnx>=1.16", "onnxruntime",
        "onnxruntime-gpu", "tritonclient[grpc,http]", "matplotlib", "tqdm"],
    7: ["torch>=2.2", "transformers>=4.40", "vllm", "fastapi", "uvicorn",
        "tritonclient[grpc,http]", "pydantic>=2", "tqdm"],
    8: ["fastapi", "uvicorn[standard]", "python-multipart", "pydantic>=2",
        "httpx", "tritonclient[grpc,http]", "matplotlib"],
    9: ["torch>=2.2", "transformers>=4.40", "fastapi", "uvicorn",
        "tritonclient[grpc,http]", "vllm", "matplotlib", "tqdm"],
    10: ["matplotlib", "ipywidgets"],
}

SESSION_AGENDA: dict[int, str] = {
    1: "- DocLayout-YOLO + DB-Net + MixNet detection benchmark on SROIE.\n"
       "- Output: `detection_benchmark.ipynb`, MixNet checkpoint, F1 table.\n",
    2: "(See full content in `02_ocr_finetune_parseq.ipynb`.)\n",
    3: "- LayoutLMv3 fine-tune for KIE on SROIE.\n"
       "- Input: OCR JSON from S2. Output: `layoutlmv3_sroie_kie.bin`, F1 per field.\n",
    4: "- Qwen2.5-3B zero-shot KIE eval; QLoRA fine-tune only if Macro F1 < 0.85.\n"
       "- Output: zero-shot report; LoRA adapter (conditional).\n",
    5: "- VLM theory (GLM-OCR, LightonOCR, Nanonets-OCR) + zero-shot demo.\n"
       "- No fine-tune; report trade-offs vs traditional pipeline.\n",
    6: "- Export PARSeq + LayoutLMv3 to ONNX; deploy on Triton; benchmark latency.\n",
    7: "- Triton Ensemble DAG + vLLM serve Qwen2.5-3B; FastAPI skeleton.\n",
    8: "- FastAPI POST /kie?backend={layoutlmv3|qwen}; throughput benchmark; Swagger demo.\n",
    9: "- Capstone E2E: Layout → OCR → KIE → FastAPI; F1 + latency + throughput report.\n",
    10: "- Demo, presentation, Q&A, roadmap.\n",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True, choices=range(1, 11))
    args = ap.parse_args()
    fname = SESSION_FILENAMES[args.session]
    out = Path("notebooks") / fname
    build_stub(args.session, SESSION_DEPS[args.session],
               SESSION_AGENDA[args.session], out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
