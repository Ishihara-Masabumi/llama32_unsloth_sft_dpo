"""Reasoning eval セットでアキュラシーを計測。

eval.jsonl の各行から、user 入力で生成させ、生成文中の [ANSWER] X を gold(assistantの[ANSWER]X)と比較。
データセット別 (gsm8k / aqua_rat / hotpotqa / strategyqa) の正解率と Overall を出力。
"""
from __future__ import annotations

import unsloth  # noqa: F401
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

MAX_SEQ_LEN = 1536


def extract_answer(text: str) -> str | None:
    m = re.search(r"\[ANSWER\]\s*([^\n]+)", text)
    if not m:
        return None
    return m.group(1).strip()


def normalize(ans: str, source: str) -> str:
    a = ans.strip().rstrip(".").strip()
    if source == "strategyqa":
        return "yes" if a.lower().startswith("y") else "no"
    if source == "aqua_rat":
        m = re.match(r"^([A-E])", a)
        return m.group(1) if m else a.upper()
    if source == "gsm8k":
        # 数値のみに正規化
        m = re.search(r"-?[\d.,]+", a)
        return m.group(0).replace(",", "") if m else a
    return a.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter if args.adapter else args.model,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    rows = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    per_source_total = defaultdict(int)
    per_source_correct = defaultdict(int)
    details = []
    for i, r in enumerate(rows):
        src = r["source"]
        gold_full = r["messages"][1]["content"]
        gold_ans = extract_answer(gold_full) or ""
        user_text = r["messages"][0]["content"]
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred_ans = extract_answer(gen) or ""
        ok = normalize(pred_ans, src) == normalize(gold_ans, src)
        per_source_total[src] += 1
        if ok:
            per_source_correct[src] += 1
        details.append({"i": i, "source": src, "gold": gold_ans, "pred": pred_ans, "ok": ok})
        if (i + 1) % 50 == 0:
            done = sum(per_source_correct.values())
            tot = sum(per_source_total.values())
            print(f"[{i+1}/{len(rows)}] overall={done/tot:.4f}")

    per_source = {
        k: per_source_correct[k] / max(1, per_source_total[k]) for k in per_source_total
    }
    overall = sum(per_source_correct.values()) / max(1, sum(per_source_total.values()))
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "n": len(rows),
        "overall_accuracy": overall,
        "per_source": per_source,
        "per_source_total": dict(per_source_total),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
