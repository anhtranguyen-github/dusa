"""Replace S1 pip-install cell(s) with the 3-cell uv pattern. Idempotent."""
import json
from pathlib import Path

P = Path("notebooks/01_detection_benchmark.ipynb")
nb = json.loads(P.read_text())

UV_REQ = """\
%%writefile requirements/buoi-1.txt
torch>=2.2
torchvision
transformers>=4.40
ultralytics
shapely
pyclipper
python-doctr[torch]
opencv-python
pillow
numpy
pyyaml
matplotlib
tqdm
timm
editdistance
"""

UV_INSTALL = """\
import shutil, subprocess, sys
if shutil.which("uv") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
subprocess.check_call(["uv", "pip", "install", "--system", "-q",
                       "-r", "requirements/buoi-1.txt"])
"""

REQ_CELL = {"cell_type": "code", "id": "s1-req", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": UV_REQ.splitlines(keepends=True)}
UV_CELL = {"cell_type": "code", "id": "s1-uv", "metadata": {},
           "execution_count": None, "outputs": [],
           "source": UV_INSTALL.splitlines(keepends=True)}

# Idempotency: skip if already patched (look for our cell IDs).
existing_ids = {c.get("id") for c in nb["cells"]}
if "s1-req" in existing_ids or "s1-uv" in existing_ids:
    print(f"already patched, nothing to do ({len(nb['cells'])} cells)")
    raise SystemExit(0)

# Insert the two uv cells right after the title (cell 0).
new_cells = [nb["cells"][0], REQ_CELL, UV_CELL] + nb["cells"][1:]

nb["cells"] = new_cells
P.write_text(json.dumps(nb, indent=1) + "\n")
print(f"patched {P} ({len(new_cells)} cells)")
