"""S1 — MixNet fine-tune on SROIE for text detection.

⚠ MixNet for text detection is research-grade and not pip-installable. This
script lays out the training loop, augmentation, and DBNet-style supervision
so that once you vendor the MixNet architecture (under
`src/detection/mixnet_arch.py`), training works with a one-line model swap.

Reference architectures to vendor:
  - Backbone: MixNet-{S,M,L} from `timm` (search `timm.create_model("mixnet_l")`).
  - Head: DBNet differentiable-binarization head (2-channel: prob + thresh).
  - Loss: BCE on prob map + Dice on shrunk-polygon mask + L1 on threshold band.

For a working detector now, use `src.detection.run_benchmark` with the DBNet
pretrained baseline. After fine-tuning here, drop the checkpoint at
`checkpoints/detection/mixnet_sroie_finetuned.pth` and the benchmark picks it
up via `MixNetDetector(checkpoint=...)`.

Usage:
    python -m src.detection.train_mixnet --config configs/detection/mixnet_sroie.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _build_backbone(cfg: dict):
    """timm gives us a ready-made MixNet backbone with ImageNet pretraining."""
    try:
        import timm
    except ImportError as e:
        raise SystemExit(
            "timm is required for the MixNet backbone. Install with: pip install timm"
        ) from e
    backbone = timm.create_model(
        cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained_backbone"] == "timm",
        features_only=True,
        out_indices=(1, 2, 3, 4),
    )
    return backbone


def _build_model(cfg: dict):
    """Stitch backbone + DB-style head. The head implementation is left as a
    follow-up exercise — students vendor it from the MixNet paper repo or
    adapt the doctr DBNet head under doctr.models.detection.differentiable_binarization.
    """
    raise NotImplementedError(
        "MixNet detection head is not vendored yet. Two options:\n"
        "  1. Copy doctr's DBNet head (doctr/models/detection/differentiable_binarization/"
        "pytorch.py — class DBHead) and use the MixNet backbone above as feature extractor.\n"
        "  2. Vendor the official MixNet text-detection repo under src/detection/mixnet_arch.py."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from .dataset_sroie import SROIEDetectionDataset

    # Demonstrate the pipeline is wired by loading one batch (no train yet).
    train_ds = SROIEDetectionDataset(
        split=cfg["data"]["splits"]["train"],
        image_size=cfg["data"]["image_height"],
        augment=True,
        min_text_size=cfg["data"]["min_text_size"],
    )
    print(f"train dataset: {len(train_ds)} receipts")
    sample = train_ds[0]
    print(f"sample image shape={tuple(sample['image'].shape)} "
          f"prob_map shape={tuple(sample['prob_map'].shape)} "
          f"text-pixel ratio={float(sample['prob_map'].mean()):.4f}")

    # Backbone is real — proves the timm install works.
    backbone = _build_backbone(cfg)
    print(f"backbone: {cfg['model']['backbone']}  "
          f"feature dims: {[f for f in backbone.feature_info.channels()]}")

    # Full model needs the head — see _build_model docstring.
    _ = _build_model(cfg)


if __name__ == "__main__":
    main()
