"""S3 — Fine-tune LayoutLMv3 for SROIE KIE.

Uses the standard HuggingFace LayoutLMv3Processor + Trainer pipeline:
  - LayoutLMv3Processor normalizes words+boxes (→ [0,1000]) and the image
    (→ 224×224 with OCR-free image features). We pass `apply_ocr=False` since
    we already have words+boxes from prep_kie.py.
  - Trainer reports loss + accuracy + per-tag F1 (seqeval).

Usage:
    python -m src.kie.layoutlm.train --config configs/kie/layoutlmv3_sroie.yaml \
        [--max_steps 50]   # smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from datasets import Dataset
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .dataset import load_image, load_split

ROOT = Path(__file__).resolve().parents[3]


def build_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name, apply_ocr=False)


def make_encode_fn(processor, label_list: list[str], max_length: int):
    label2id = {l: i for i, l in enumerate(label_list)}
    pad_id = label2id["O"]

    def encode(batch):
        images = [load_image(p, ROOT) for p in batch["image_path"]]
        words = batch["words"]
        boxes = batch["boxes"]
        word_labels = batch["ner_tags"]
        enc = processor(
            images=images,
            text=words,
            boxes=boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v for k, v in enc.items()}

    return encode, pad_id


def compute_metrics_factory(label_list: list[str]):
    id2label = {i: l for i, l in enumerate(label_list)}

    def fn(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=2)
        true_preds, true_labels = [], []
        for p_row, l_row in zip(preds, labels):
            tp, tl = [], []
            for pi, li in zip(p_row, l_row):
                if li == -100:
                    continue
                tp.append(id2label[int(pi)])
                tl.append(id2label[int(li)])
            true_preds.append(tp)
            true_labels.append(tl)
        return {
            "accuracy": accuracy_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds, zero_division=0),
            "recall": recall_score(true_labels, true_preds, zero_division=0),
            "f1_macro": f1_score(true_labels, true_preds, average="macro", zero_division=0),
            "f1_micro": f1_score(true_labels, true_preds, average="micro", zero_division=0),
        }
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prepared = ROOT / cfg["data"]["prepared_dir"]
    label_list = json.loads((prepared / cfg["data"]["label_list"]).read_text())
    print(f"labels: {label_list}")

    train_ds = load_split(prepared, cfg["data"]["train_jsonl"])
    test_ds = load_split(prepared, cfg["data"]["test_jsonl"])
    print(f"train: {len(train_ds)}  test: {len(test_ds)}")

    processor = build_processor(cfg["model"]["pretrained"])
    encode, _ = make_encode_fn(processor, label_list, cfg["model"]["max_length"])
    train_enc = train_ds.map(encode, batched=True, batch_size=8,
                             remove_columns=train_ds.column_names)
    test_enc = test_ds.map(encode, batched=True, batch_size=8,
                           remove_columns=test_ds.column_names)
    print(f"encoded; example keys: {list(train_enc[0].keys())}")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        cfg["model"]["pretrained"],
        num_labels=len(label_list),
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
    )

    out_dir = ROOT / cfg["train"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": cfg["train"]["per_device_train_batch_size"],
        "per_device_eval_batch_size": cfg["train"]["per_device_eval_batch_size"],
        "learning_rate": cfg["train"]["learning_rate"],
        "num_train_epochs": cfg["train"]["num_train_epochs"],
        "weight_decay": cfg["train"]["weight_decay"],
        "warmup_ratio": cfg["train"]["warmup_ratio"],
        "logging_steps": cfg["train"]["logging_steps"],
        "load_best_model_at_end": cfg["train"]["load_best_model_at_end"],
        "metric_for_best_model": cfg["train"]["metric_for_best_model"],
        "greater_is_better": cfg["train"]["greater_is_better"],
        "fp16": cfg["train"]["fp16"],
        "label_smoothing_factor": cfg["train"]["label_smoothing_factor"],
        "report_to": [],
    }
    # transformers renamed `evaluation_strategy` → `eval_strategy` at 4.40
    try:
        args_obj = TrainingArguments(
            **train_kwargs,
            eval_strategy=cfg["train"]["eval_strategy"],
            save_strategy=cfg["train"]["save_strategy"],
            max_steps=args.max_steps or -1,
        )
    except TypeError:
        args_obj = TrainingArguments(
            **train_kwargs,
            evaluation_strategy=cfg["train"]["eval_strategy"],
            save_strategy=cfg["train"]["save_strategy"],
            max_steps=args.max_steps or -1,
        )

    trainer = Trainer(
        model=model,
        args=args_obj,
        train_dataset=train_enc,
        eval_dataset=test_enc,
        tokenizer=processor,
        compute_metrics=compute_metrics_factory(label_list),
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("final metrics:", metrics)

    alias = out_dir / cfg["checkpoint_alias"]
    trainer.save_model(str(out_dir))
    # also save a single-file checkpoint for compatibility with the curriculum naming
    import torch
    torch.save(model.state_dict(), alias)
    print(f"✓ checkpoint saved: {alias}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
