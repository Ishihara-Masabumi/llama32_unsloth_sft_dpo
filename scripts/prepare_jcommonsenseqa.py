"""JCommonsenseQA を SFT 用 chat フォーマットに整形して JSONL で保存。

二種のフォーマット:
  - letter_text  : assistant が "A. <選択肢テキスト>" 形式で答える
  - answer_tag   : assistant が "[ANSWER] A" 形式で答える
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

LETTERS = ["A", "B", "C", "D", "E"]


def build_user_prompt(row: dict) -> str:
    choices = [row[f"choice{i}"] for i in range(5)]
    body = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        "次の日本語の常識問題に答えてください。A〜Eの選択肢の中から最も適切なものを1つ選びなさい。\n\n"
        f"問題: {row['question']}\n\n"
        f"選択肢:\n{body}"
    )


def build_assistant(row: dict, fmt: str) -> str:
    label = int(row["label"])
    letter = LETTERS[label]
    choice_text = row[f"choice{label}"]
    if fmt == "letter_text":
        return f"{letter}. {choice_text}"
    if fmt == "answer_tag":
        return f"[ANSWER] {letter}"
    raise ValueError(fmt)


def to_messages(row: dict, fmt: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": build_user_prompt(row)},
            {"role": "assistant", "content": build_assistant(row, fmt)},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sbintuitions/JCommonsenseQA")
    ap.add_argument("--out-dir", default="data/jcommonsenseqa")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    for split in ("train", "validation"):
        for fmt in ("letter_text", "answer_tag"):
            path = out_dir / f"{split}.{fmt}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in ds[split]:
                    f.write(json.dumps(to_messages(row, fmt), ensure_ascii=False) + "\n")
            print(f"[OK] wrote {path} ({len(ds[split])} rows)")

    # 評価用に raw も保存(検証時に label / choices を直接見るため)
    raw_path = out_dir / "validation.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for row in ds["validation"]:
            f.write(json.dumps({
                "question": row["question"],
                "choices": [row[f"choice{i}"] for i in range(5)],
                "label": int(row["label"]),
            }, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {raw_path}")


if __name__ == "__main__":
    main()
