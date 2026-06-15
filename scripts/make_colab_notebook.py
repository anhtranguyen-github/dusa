"""One-shot: build notebooks/01_detection_benchmark_colab.ipynb from the local variant.

Replaces the local setup cells with Colab bootstrap (Drive mount, repo clone,
pip install, SROIE download), bumps MixNet batch to 8 (Colab T4 has 15 GB VRAM),
and clears cell outputs so the published notebook is clean.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

SRC = Path("notebooks/01_detection_benchmark.ipynb")
DST = Path("notebooks/colab/01_detection_benchmark.ipynb")

REPO_URL = "https://github.com/YOUR_USER/dusa.git"  # ← edit before publishing


def mkmd(cid: str, src: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src.splitlines(keepends=True)}


def mkcode(cid: str, src: str) -> dict:
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


COLAB_HEADER = mkmd("colab-header", f"""# Buổi 1 — Layout & Text Detection (Colab)

<a href="https://colab.research.google.com/github/YOUR_USER/dusa/blob/main/notebooks/colab/01_detection_benchmark.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Run order:
1. **Runtime → Change runtime type → GPU** (T4 or better).
2. Run the four bootstrap cells below (mount Drive, clone repo, install deps, download SROIE) — ~3-5 min.
3. Then run the rest of the notebook top-to-bottom as in the local version.

> ⚙ Edit `REPO_URL` in the clone cell if your fork lives elsewhere.

## Bootstrap (Colab only)
""")


COLAB_DRIVE = mkcode("colab-drive", """# (Optional) mount Drive so checkpoints + figures persist across Colab sessions.
# Skip this cell if you only need a one-off run.
from google.colab import drive
drive.mount('/content/drive')
""")


COLAB_CLONE = mkcode("colab-clone", f"""import os, subprocess
REPO_URL = "{REPO_URL}"  # ← edit if your fork is elsewhere
REPO_DIR = "/content/dusa"

if not os.path.isdir(REPO_DIR):
    print('cloning', REPO_URL)
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR], check=True)
else:
    print('pulling latest in', REPO_DIR)
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'], check=True)
%cd $REPO_DIR
!ls
""")


COLAB_PIP = mkcode("colab-pip", """# Colab ships torch + matplotlib + pandas + opencv; install the rest.
%pip install -q timm shapely pyclipper python-doctr[torch] datasets editdistance
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
""")


COLAB_DATA = mkcode("colab-data", """# Download SROIE (≈30 s; 522 train + 130 test receipts).
import os
if not os.path.isdir('data/sroie/images/test'):
    !python scripts/download_sroie.py
else:
    print('SROIE already present')
!ls data/sroie/images/train | head -3 && echo ... && ls data/sroie/images/test | wc -l
""")


COLAB_IMPORTS = mkcode("colab-imports", """# Same imports as the local notebook — re-uses src/ now that we cd'd into the repo.
import os, sys
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
print('cwd:', PROJECT_ROOT)

from src.data.sroie_loader import list_ids, iter_split
from src.detection import DBNetDetector, MixNetDetector, evaluate_image, aggregate, draw_overlay
print('SROIE test images available:', len(list_ids('test')))
""")


def main() -> None:
    nb = json.loads(SRC.read_text())

    # Drop the two local-setup cells: 6712b6b4 (path setup) and 58e0617f (imports).
    LOCAL_SETUP_IDS = {"6712b6b4", "58e0617f"}
    kept = [c for c in nb["cells"] if c.get("id") not in LOCAL_SETUP_IDS]

    # Replace the original top-level title (cell 00319509) with the Colab header.
    new_cells = []
    for c in kept:
        if c.get("id") == "00319509":
            # Replace with Colab header + bootstrap (Drive optional, clone, pip, data, imports).
            new_cells.extend([COLAB_HEADER, COLAB_DRIVE, COLAB_CLONE, COLAB_PIP, COLAB_DATA, COLAB_IMPORTS])
        else:
            new_cells.append(c)

    # Clear outputs + execution counts (clean publish state).
    for c in new_cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # Bump MixNet batch_size to 8 in the fine-tune cell (Colab T4 has 15 GB).
    for c in new_cells:
        if c.get("id") == "05f3226e":
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            src = src.replace(
                "'mixnet_finetuned': {'backbone': 'mixnet_l',              'batch_size': 4, 'short': 'mixnet'},",
                "'mixnet_finetuned': {'backbone': 'mixnet_l',              'batch_size': 8, 'short': 'mixnet'},",
            )
            src = src.replace(
                "- **MixNet side** — `mixnet_l`, batch 4 (8 GB VRAM doesn't fit batch 8 at 736²).",
                "- **MixNet side** — `mixnet_l`, batch 8 (Colab T4/L4 ≥15 GB fits comfortably).",
            )
            c["source"] = src.splitlines(keepends=True)

    # Drop the local-only "next steps" stale wording about checkpoints living locally.
    nb["cells"] = new_cells
    nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {DST}  ({len(new_cells)} cells)")


if __name__ == "__main__":
    main()
