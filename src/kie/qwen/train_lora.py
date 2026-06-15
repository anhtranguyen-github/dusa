"""S4 BƯỚC 2 — QLoRA fine-tune Qwen2.5-3B-Instruct on SROIE KIE.

Only runs when zero-shot Macro F1 < 0.85 (see zeroshot.py). Loads the chat-
formatted JSONL produced by `src.data.prep_instructions` and fine-tunes a
LoRA adapter via TRL's SFTTrainer in 4-bit (QLoRA).

Usage:
    python -m src.kie.qwen.train_lora --config configs/training/qwen_qlora.yaml \
        [--max_steps 10]   # smoke test

Requires (GPU): bitsandbytes, peft, trl, accelerate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Lazy imports — these are heavy and not needed if students skip QLoRA.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    prepared = ROOT / cfg["data"]["prepared_dir"]
    train_path = prepared / cfg["data"]["train_jsonl"]
    test_path = prepared / cfg["data"]["test_jsonl"]

    print(f"Loading dataset from {train_path}")
    ds = load_dataset("json", data_files={
        "train": str(train_path),
        "test": str(test_path),
    })

    tok = AutoTokenizer.from_pretrained(
        cfg["model"]["pretrained"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    quant_cfg = None
    if cfg["train"]["load_4bit"]:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading {cfg['model']['pretrained']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["pretrained"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
    )

    out_dir = ROOT / cfg["train"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        logging_steps=cfg["train"]["logging_steps"],
        save_strategy=cfg["train"]["save_strategy"],
        bf16=cfg["train"]["bf16"],
        max_steps=args.max_steps or -1,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"✓ LoRA adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
