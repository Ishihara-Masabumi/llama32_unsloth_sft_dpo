"""DPO の eval セット (784件) で chosen 選好精度を測定。

各 (prompt, chosen, rejected) について、モデルが chosen に対して与える対数尤度が
rejected の対数尤度より大きい比率を Chosen 選好精度とする。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MAX_SEQ_LEN = 1024


def conditional_logprob(model, tokenizer, prompt: str, response: str) -> float:
    messages_prompt = [{"role": "user", "content": prompt}]
    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    full_messages = messages_prompt + [{"role": "assistant", "content": response}]
    full_ids = tokenizer.apply_chat_template(
        full_messages, add_generation_prompt=False, return_tensors="pt"
    ).to(model.device)
    if full_ids.shape[1] <= prompt_ids.shape[1]:
        return float("-inf")
    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # response 部分のみ
    start = prompt_ids.shape[1] - 1
    resp_lp = token_lp[0, start:]
    return float(resp_lp.sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--output", required=True)
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

    correct = 0
    details = []
    for i, r in enumerate(rows):
        lp_c = conditional_logprob(model, tokenizer, r["prompt"], r["chosen"])
        lp_r = conditional_logprob(model, tokenizer, r["prompt"], r["rejected"])
        win = lp_c > lp_r
        if win:
            correct += 1
        details.append({"i": i, "lp_chosen": lp_c, "lp_rejected": lp_r, "chosen_preferred": win})
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(rows)}] chosen_pref_acc={correct/(i+1):.4f}")

    acc = correct / len(rows)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "n": len(rows),
        "chosen_preference_accuracy": acc,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
