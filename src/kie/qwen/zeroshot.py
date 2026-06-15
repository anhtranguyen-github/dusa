"""S4 BƯỚC 1 — Zero-shot evaluation of Qwen2.5-3B on SROIE KIE.

The curriculum gate: if Macro F1 ≥ 0.85 on SROIE test, **skip fine-tuning** and
record the result. Otherwise proceed to train_lora.py.

Usage:
    python -m src.kie.qwen.zeroshot --config configs/training/qwen_qlora.yaml \
        [--sample_limit 20]   # quick CPU sanity run

Outputs:
  reports/benchmarks/qwen_zeroshot_results.json
        { macro_f1, per_field, exact_match_rate, predictions, gate_passed }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from .metrics import evaluate, parse_json_output

ROOT = Path(__file__).resolve().parents[3]


def _load_messages(jsonl_path: Path) -> list[dict]:
    out = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_pipeline(model_name: str, trust_remote_code: bool):
    """Returns a generate(messages) → text function. Lazy-imports torch+transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} …")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    device = next(model.parameters()).device

    def generate(messages, max_new_tokens: int, temperature: float) -> str:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        full = tok.decode(out[0], skip_special_tokens=True)
        return full[len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample_limit", type=int, default=None,
                    help="Override config sample_limit (e.g., 20 for a smoke test).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    zs = cfg["zero_shot"]
    sample_limit = args.sample_limit if args.sample_limit is not None else zs["sample_limit"]

    prepared = ROOT / cfg["data"]["prepared_dir"]
    test_records = _load_messages(prepared / cfg["data"]["test_jsonl"])
    targets = json.loads((prepared / cfg["data"]["targets"]).read_text())

    if sample_limit:
        test_records = test_records[:sample_limit]

    generate = _build_pipeline(cfg["model"]["pretrained"], cfg["model"]["trust_remote_code"])

    predictions: dict[str, dict] = {}
    t0 = time.time()
    for i, rec in enumerate(test_records):
        rid = rec["id"]
        # drop the assistant turn so the model has to produce it
        user_messages = [m for m in rec["messages"] if m["role"] != "assistant"]
        raw = generate(user_messages, zs["max_new_tokens"], zs["temperature"])
        predictions[rid] = parse_json_output(raw)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(test_records)}  ({(time.time()-t0)/(i+1):.1f}s/it)")

    result = evaluate(predictions, targets)
    gate = result.macro_f1 >= zs["f1_threshold"]
    out_dir = ROOT / zs["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / zs["result_filename"]
    payload = {
        "model": cfg["model"]["pretrained"],
        "n_evaluated": result.n,
        "macro_f1": result.macro_f1,
        "exact_match_rate": result.exact_match_rate,
        "per_field": result.per_field,
        "f1_threshold": zs["f1_threshold"],
        "gate_passed_skip_finetune": gate,
        "predictions": predictions,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print(f"Macro F1 = {result.macro_f1:.4f}  (threshold {zs['f1_threshold']})")
    print(f"gate {'PASSED — skip fine-tune' if gate else 'FAILED — proceed to QLoRA'}")
    print(f"✓ {out_path}")


if __name__ == "__main__":
    main()
