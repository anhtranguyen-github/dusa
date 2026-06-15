import nbformat

from scripts.make_session_stub import build_stub


def test_stub_has_three_uv_cells(tmp_path):
    out = tmp_path / "03_kie.ipynb"
    build_stub(session=3, requirements_lines=["transformers>=4.40", "seqeval"],
               agenda_md="Agenda for session 3.", out_path=out)
    nb = nbformat.read(out, as_version=4)
    sources = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
               for c in nb.cells]
    assert nb.cells[0]["cell_type"] == "markdown"
    assert "Buổi 3" in sources[0]
    assert "%%writefile requirements/buoi-3.txt" in sources[1]
    assert "uv" in sources[2] and "pip" in sources[2] and "--system" in sources[2]
    assert "Agenda for session 3" in sources[3]
