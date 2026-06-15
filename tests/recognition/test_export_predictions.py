import json
from pathlib import Path

import numpy as np

from src.recognition.export_predictions import build_ocr_json


class _StubAdapter:
    name = "stub"

    def predict(self, crops):
        return [f"line{i}" for i in range(len(crops))]


def _fake_image(h=40, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_build_ocr_json_groups_lines_per_image_id(tmp_path):
    examples = [
        ("img1", _fake_image(), [
            {"polygon": [(0, 0), (10, 0), (10, 10), (0, 10)], "text": "GT1"},
            {"polygon": [(0, 20), (10, 20), (10, 30), (0, 30)], "text": "GT2"},
        ]),
        ("img2", _fake_image(), [
            {"polygon": [(0, 0), (20, 0), (20, 10), (0, 10)], "text": "GT3"},
        ]),
    ]
    out_path = tmp_path / "ocr.json"
    n = build_ocr_json(examples, _StubAdapter(), out_path)
    assert n == 3
    data = json.loads(out_path.read_text())
    assert set(data.keys()) == {"img1", "img2"}
    assert len(data["img1"]) == 2
    assert data["img1"][0]["text"].startswith("line")
    assert data["img1"][0]["bbox"] == [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert "conf" in data["img1"][0]
