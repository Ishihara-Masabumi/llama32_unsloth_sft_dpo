"""Reasoning SFT(<thinking>...</thinking>\n[ANSWER] X 形式)を unsloth+QLoRA で学習。
"""
from __future__ import annotations

import unsloth  # noqa: F401  IMPORTANT: must be imported before trl/transformers/peft
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

MAX_SEQ_LEN = 1536


def load_jsonl(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="既存 LoRA アダプタ (MCQ-SFT) から続けて学習")
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_name = args.adapter if args.adapter else args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    ds = load_jsonl(args.train_jsonl)

    def fmt(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    eos_tok = "<|eot_id|>" if "<|eot_id|>" in tokenizer.get_vocab() else tokenizer.eos_token
    pad_tok = tokenizer.pad_token if tokenizer.pad_token else eos_tok
    print(f"[info] using eos_token={eos_tok!r}, pad_token={pad_tok!r}")

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=args.seed,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
        eos_token=eos_tok,
        pad_token=pad_tok,
    )

    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds, args=cfg)
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "train_meta.json").write_text(json.dumps({
        "base_model": args.model,
        "warmstart_adapter": args.adapter,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
    }, ensure_ascii=False, indent=2))
    print(f"[OK] Reasoning SFT done -> {args.output_dir}")


if __name__ == "__main__":
    main()
