"""JCommonsenseQA validation でモデルを評価。

ベース、または LoRA アダプタを当てたモデルで実行可能。
"""
from __future__ import annotations

import unsloth  # noqa: F401
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

import argparse
import json
import re
from pathlib import Path

import torch

LETTERS = ["A", "B", "C", "D", "E"]
MAX_SEQ_LEN = 1024


def build_user_prompt(row: dict) -> str:
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return (
        "次の日本語の常識問題に答えてください。A〜Eの選択肢の中から最も適切なものを1つ選びなさい。\n\n"
        f"問題: {row['question']}\n\n"
        f"選択肢:\n{body}"
    )


def parse_answer(text: str, strict: str = "lenient") -> str | None:
    """生成テキストから A〜E のいずれかを取り出す。

    strict:
      - "lenient" (デフォルト) : [ANSWER] X > 行頭 X. > 単独 X
      - "answer_tag"           : [ANSWER] X のみ
      - "letter_text"          : 行頭 X. (またはテキスト先頭の X) のみ
    """
    if strict in ("lenient", "answer_tag"):
        m = re.search(r"\[ANSWER\]\s*([A-E])", text)
        if m:
            return m.group(1)
        if strict == "answer_tag":
            return None
    if strict in ("lenient", "letter_text"):
        m = re.search(r"(?m)^\s*([A-E])[.\s]", text)
        if m:
            return m.group(1)
        # 先頭(空白を許す)で X. を試す
        m = re.match(r"\s*([A-E])\b", text)
        if m and strict == "letter_text":
            return m.group(1)
        if strict == "letter_text":
            return None
    # lenient のみ最後のフォールバック
    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory (optional)")
    ap.add_argument("--raw-jsonl", default="data/jcommonsenseqa/validation.raw.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--strict", default="lenient", choices=["lenient", "answer_tag", "letter_text"])
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
    with open(args.raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    correct = 0
    unparsable = 0
    details = []
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": build_user_prompt(row)}]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
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
        pred_letter = parse_answer(gen, strict=args.strict)
        if pred_letter is None:
            unparsable += 1
            pred = -1
        else:
            pred = LETTERS.index(pred_letter)
        gold = row["label"]
        if pred == gold:
            correct += 1
        details.append({
            "i": i,
            "gold": gold,
            "pred": pred,
            "pred_letter": pred_letter,
            "gen": gen,
        })
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(rows)}] acc={correct/(i+1):.4f} unparsable={unparsable}")

    acc = correct / len(rows)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "strict": args.strict,
        "n": len(rows),
        "correct": correct,
        "accuracy": acc,
        "unparsable": unparsable,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
