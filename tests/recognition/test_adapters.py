import numpy as np
import pytest

from src.recognition.adapters import RecogAdapter


class _DummyAdapter(RecogAdapter):
    name = "dummy"

    def __init__(self):
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        return ["stub"] * len(images)


def test_adapter_protocol_predict_returns_one_string_per_image():
    a = _DummyAdapter()
    images = [np.zeros((32, 128, 3), dtype=np.uint8) for _ in range(3)]
    out = a.predict(images)
    assert out == ["stub", "stub", "stub"]
    assert a.calls == 1
