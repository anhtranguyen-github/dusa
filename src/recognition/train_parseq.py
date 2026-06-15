"""S2 — PARSeq fine-tune on SROIE line crops.

Loads `baudm/parseq-tiny` (or any HF-mirrored PARSeq) and fine-tunes on the
crops produced by `src/data/prep_recognition.py`. Reports CER/WER on the test
split each epoch.

This script keeps the boilerplate minimal so the lab can focus on the model;
the official baudm/parseq repo offers a richer Lightning recipe — clone it for
the multi-orientation / cosine-restart training used in the paper.

Usage:
    python -m src.recognition.train_parseq \
        --config configs/recognition/parseq_sroie.yaml \
        [--max_steps 100]   # for a CPU smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]


def _build_loaders(cfg: dict):
    from .dataset import SROIERecognitionDataset
    d = cfg["data"]
    train_ds = SROIERecognitionDataset(
        d["root"], d["train_labels"],
        image_height=d["image_height"], image_width=d["image_width"],
        augment=True, max_label_length=d["max_label_length"],
    )
    test_ds = SROIERecognitionDataset(
        d["root"], d["test_labels"],
        image_height=d["image_height"], image_width=d["image_width"],
        augment=False, max_label_length=d["max_label_length"],
    )
    bs = cfg["train"]["batch_size"]
    nw = d["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, collate_fn=_collate)
    return train_loader, test_loader


def _collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    texts = [b[1] for b in batch]
    return images, texts


def _load_model(cfg: dict):
    """Try torch.hub (official baudm route). Falls back to a clear error message
    so the user installs the upstream repo when needed.
    """
    name = cfg["model"]["pretrained"]
    try:
        # The official PARSeq repo registers torch.hub entrypoints.
        model = torch.hub.load("baudm/parseq", name.split("/")[-1].replace("-", "_"),
                               pretrained=True, trust_repo=True)
    except Exception as e:
        raise SystemExit(
            "Could not load PARSeq via torch.hub. Install the upstream repo:\n"
            "    pip install 'git+https://github.com/baudm/parseq.git'\n"
            f"Original error: {e}"
        )
    return model


def _cer(pred: str, gt: str) -> float:
    import editdistance
    return editdistance.eval(pred, gt) / max(len(gt), 1)


def _wer(pred: str, gt: str) -> float:
    import editdistance
    return editdistance.eval(pred.split(), gt.split()) / max(len(gt.split()), 1)


def evaluate(model, loader, device, log_first_n: int = 0) -> dict:
    model.eval()
    cers, wers = [], []
    n_logged = 0
    with torch.no_grad():
        for images, texts in loader:
            images = images.to(device)
            logits = model(images)
            pred_strs = model.tokenizer.decode(logits)[0] if hasattr(model, "tokenizer") \
                else _greedy_decode(logits)
            for p, g in zip(pred_strs, texts):
                cers.append(_cer(p, g))
                wers.append(_wer(p, g))
                if n_logged < log_first_n:
                    print(f"  gt={g!r:40s} pred={p!r}")
                    n_logged += 1
    return {"cer": sum(cers) / len(cers) if cers else 0.0,
            "wer": sum(wers) / len(wers) if wers else 0.0}


def _greedy_decode(logits) -> list[str]:
    # Generic fallback if model doesn't ship a tokenizer attribute.
    ids = logits.argmax(dim=-1).cpu().tolist()
    return ["".join(map(str, row)) for row in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Cap training steps for smoke tests.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, test_loader = _build_loaders(cfg)
    print(f"train batches: {len(train_loader)}  test batches: {len(test_loader)}")

    model = _load_model(cfg).to(device)
    if cfg["model"].get("freeze_backbone"):
        for p in getattr(model, "encoder", []).parameters():
            p.requires_grad = False

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    step = 0
    for epoch in range(cfg["train"]["max_epochs"]):
        model.train()
        for images, texts in train_loader:
            images = images.to(device)
            targets = model.tokenizer.encode(texts).to(device) if hasattr(model, "tokenizer") \
                else torch.zeros((len(texts),), device=device)
            logits = model(images)
            loss = model.loss(logits, targets) if hasattr(model, "loss") \
                else torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                cfg["train"]["grad_clip_norm"],
            )
            optim.step()
            step += 1
            if step % 50 == 0:
                print(f"  step {step}  loss={loss.item():.4f}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        metrics = evaluate(model, test_loader, device,
                           log_first_n=cfg["eval"]["log_first_n_predictions"])
        print(f"epoch {epoch}: CER={metrics['cer']:.4f}  WER={metrics['wer']:.4f}")

    out = ROOT / cfg["checkpoint"]["output_dir"] / cfg["checkpoint"]["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"✓ saved {out}")

    metrics_out = out.parent / "parseq_sroie_metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2))
    print(f"✓ metrics {metrics_out}")


if __name__ == "__main__":
    main()
