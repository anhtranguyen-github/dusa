"""Generate the Colab variant of a session notebook.

Usage:
    python scripts/make_colab_notebook.py --session 1
    python scripts/make_colab_notebook.py --session 2

The local notebook stays the source of truth; this script prepends a Colab
bootstrap (install uv, clone repo, cd) and clears outputs. All session
notebooks use the same 3-cell uv install pattern internally, so the bootstrap
is uniform.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_URL = "https://github.com/anhtranguyen-github/dusa.git"

SESSION_TITLES = {
    1: "Buổi 1 — Layout & Text Detection",
    2: "Buổi 2 — Text Recognition (OCR)",
    3: "Buổi 3 — KIE: LayoutLMv3 Fine-tune",
    4: "Buổi 4 — LLM & Qwen2.5-3B KIE",
    5: "Buổi 5 — Vision-Language Models",
    6: "Buổi 6 — ONNX & Triton",
    7: "Buổi 7 — Triton Ensemble + vLLM + FastAPI (1)",
    8: "Buổi 8 — FastAPI Dual-Backend (2)",
    9: "Buổi 9 — Capstone E2E",
    10: "Buổi 10 — Demo & Wrap-up",
}

SESSION_FILENAMES = {
    1: "01_detection_benchmark.ipynb",
    2: "02_ocr_finetune_parseq.ipynb",
    3: "03_kie_layoutlmv3_finetune.ipynb",
    4: "04_qwen_kie_zeroshot_and_qlora.ipynb",
    5: "05_vlm_zeroshot_demo.ipynb",
    6: "06_onnx_triton_export.ipynb",
    7: "07_triton_ensemble_vllm.ipynb",
    8: "08_fastapi_dual_backend.ipynb",
    9: "09_capstone_e2e.ipynb",
    10: "10_demo_final.ipynb",
}


def _mkmd(cid: str, src: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True)}


def _mkcode(cid: str, src: str) -> dict:
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def _colab_header_cells(session: int, fname: str, repo_url: str) -> list[dict]:
    title = SESSION_TITLES[session]
    badge = (
        f'<a href="https://colab.research.google.com/github/anhtranguyen-github/dusa/'
        f'blob/main/notebooks/colab/{fname}" target="_parent">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
        f'alt="Open In Colab"/></a>'
    )
    header_md = (
        f"# {title} (Colab)\n\n{badge}\n\n"
        f"Runtime → Change runtime type → GPU (T4 or better), then run all cells.\n\n"
        f"## Colab bootstrap\n"
    )
    bootstrap_code = (
        "# Install uv, clone repo, cd in. Idempotent — safe to re-run.\n"
        "!pip install -q uv\n"
        f"!git clone -q {repo_url} /content/dusa || git -C /content/dusa pull --ff-only\n"
        "%cd /content/dusa\n"
        "import os; print('cwd:', os.getcwd())\n"
    )
    return [_mkmd(f"colab-h-{session}", header_md),
            _mkcode(f"colab-b-{session}", bootstrap_code)]


def build_colab_notebook(session: int, src: Path, dst: Path, repo_url: str = REPO_URL) -> None:
    """Read local notebook, prepend Colab bootstrap, clear outputs, write dst."""
    nb = json.loads(Path(src).read_text())
    bootstrap = _colab_header_cells(session, SESSION_FILENAMES[session], repo_url)
    nb["cells"] = bootstrap + nb["cells"]
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nb.setdefault("metadata", {})
    nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3",
                                    "language": "python", "name": "python3"}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True, choices=range(1, 11))
    ap.add_argument("--repo-url", default=REPO_URL)
    args = ap.parse_args()
    fname = SESSION_FILENAMES[args.session]
    src = Path("notebooks") / fname
    dst = Path("notebooks/colab") / fname
    if not src.exists():
        raise SystemExit(f"local notebook missing: {src}")
    build_colab_notebook(args.session, src, dst, args.repo_url)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
