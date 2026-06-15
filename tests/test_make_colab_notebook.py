"""Smoke: refactored generator produces a valid Colab notebook with uv bootstrap."""
import json
from pathlib import Path

import nbformat
import pytest

from scripts.make_colab_notebook import build_colab_notebook


@pytest.fixture
def local_nb(tmp_path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "header", "metadata": {},
             "source": "# Buổi 2\n"},
            {"cell_type": "code", "id": "req-write", "metadata": {},
             "execution_count": None, "outputs": [],
             "source": "%%writefile requirements/buoi-2.txt\ntorch>=2.2\n"},
            {"cell_type": "code", "id": "uv-install", "metadata": {},
             "execution_count": None, "outputs": [],
             "source": "import subprocess; subprocess.check_call(['uv','pip','install','--system','-r','requirements/buoi-2.txt'])\n"},
            {"cell_type": "code", "id": "work", "metadata": {},
             "execution_count": None, "outputs": [],
             "source": "print('hello')\n"},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                     "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    p = tmp_path / "02_local.ipynb"
    p.write_text(json.dumps(nb))
    return p


def test_build_colab_prepends_uv_bootstrap(tmp_path, local_nb):
    out = tmp_path / "02_colab.ipynb"
    build_colab_notebook(session=2, src=local_nb, dst=out,
                        repo_url="https://example.com/dusa.git")
    written = nbformat.read(out, as_version=4)
    sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
               for c in written.cells]
    assert written.cells[0]["cell_type"] == "markdown"
    assert "Colab" in sources[0]
    assert any("pip install -q uv" in s for s in sources[:5])
    assert any("git clone" in s for s in sources[:5])
    assert any("print('hello')" in s for s in sources)


def test_build_colab_clears_code_outputs(tmp_path, local_nb):
    out = tmp_path / "02_colab.ipynb"
    build_colab_notebook(session=2, src=local_nb, dst=out,
                        repo_url="https://example.com/dusa.git")
    written = nbformat.read(out, as_version=4)
    for c in written.cells:
        if c["cell_type"] == "code":
            assert c.get("outputs") == []
            assert c.get("execution_count") is None
