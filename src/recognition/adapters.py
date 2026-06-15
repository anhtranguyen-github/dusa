"""Uniform recognition-model interface for cross-model comparison.

Each adapter loads its model lazily on first predict() call to keep import-time
cheap (some weights are 300+ MB). All adapters accept HxWx3 uint8 RGB numpy
images and return a list[str].
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class RecogAdapter(Protocol):
    """Common interface: take a list of crops, return predicted strings."""
    name: str

    def predict(self, images: list[np.ndarray]) -> list[str]:
        ...


class TrOCRAdapter:
    """microsoft/trocr-base-printed, zero-shot."""
    name = "trocr-base-printed"

    def __init__(self, model_id: str = "microsoft/trocr-base-printed", device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._proc = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = TrOCRProcessor.from_pretrained(self.model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device).eval()

    def predict(self, images: list[np.ndarray]) -> list[str]:
        import torch
        from PIL import Image
        self._ensure_loaded()
        pil = [Image.fromarray(img) for img in images]
        out_texts: list[str] = []
        # Batch in chunks of 16 to bound memory on small GPUs.
        for i in range(0, len(pil), 16):
            batch = pil[i:i + 16]
            inputs = self._proc(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                ids = self._model.generate(**inputs, max_new_tokens=64)
            out_texts.extend(self._proc.batch_decode(ids, skip_special_tokens=True))
        return out_texts


class PARSeqAdapter:
    """baudm/parseq via torch.hub, optionally loading a fine-tuned state_dict."""
    name = "parseq"

    def __init__(self, hub_name: str = "parseq_tiny", checkpoint_path: str | None = None,
                 device: str | None = None):
        self.hub_name = hub_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.hub.load("baudm/parseq", self.hub_name, pretrained=True, trust_repo=True)
        except Exception as e:
            raise RuntimeError(
                "torch.hub PARSeq load failed. Install upstream:\n"
                "    uv pip install --system 'git+https://github.com/baudm/parseq.git'\n"
                f"Original: {e}"
            )
        if self.checkpoint_path is not None:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        self._model = model.to(self.device).eval()

    def predict(self, images: list[np.ndarray]) -> list[str]:
        import torch
        from PIL import Image
        from torchvision import transforms
        self._ensure_loaded()
        tf = transforms.Compose([
            transforms.Resize((32, 128), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        pil = [Image.fromarray(img).convert("RGB") for img in images]
        out: list[str] = []
        for i in range(0, len(pil), 32):
            batch = torch.stack([tf(im) for im in pil[i:i + 32]]).to(self.device)
            with torch.no_grad():
                logits = self._model(batch)
                preds = self._model.tokenizer.decode(logits)[0]
            out.extend(preds)
        return out


class CRNNAdapter:
    """CTC-decoded CRNN. Tries an HF pretrained mirror first, then falls back to
    a from-scratch tiny CRNN (`crnn_tiny.py`) trained briefly on SROIE.
    """
    name = "crnn"

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._charset = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from .crnn_tiny import CRNNTiny, DEFAULT_CHARSET
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._charset = DEFAULT_CHARSET
        self._model = CRNNTiny(num_classes=len(DEFAULT_CHARSET) + 1)
        if self.checkpoint_path is not None:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            self._model.load_state_dict(state, strict=False)
        self._model = self._model.to(self.device).eval()

    def predict(self, images: list[np.ndarray]) -> list[str]:
        import torch
        from PIL import Image
        from torchvision import transforms
        from .crnn_tiny import ctc_greedy_decode
        self._ensure_loaded()
        tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 128), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        pil = [Image.fromarray(img) for img in images]
        out: list[str] = []
        for i in range(0, len(pil), 32):
            batch = torch.stack([tf(im) for im in pil[i:i + 32]]).to(self.device)
            with torch.no_grad():
                logits = self._model(batch)
                preds = ctc_greedy_decode(logits, self._charset)
            out.extend(preds)
        return out
