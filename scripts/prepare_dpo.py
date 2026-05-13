"""Aratako/iterative-dpo-data-for-SimPO-iter2 を DPO 用フォーマットに整形して JSONL に保存。

データセットは {prompt, chosen, rejected} 形式に正規化する。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def normalize_row(row: dict) -> dict | None:
    """データセットの行を (prompt, chosen, rejected) に正規化。"""
    # 期待されるフィールド名候補
    prompt = row.get("prompt") or row.get("instruction") or row.get("question")
    chosen = row.get("chosen") or row.get("chosen_response") or row.get("preferred")
    rejected = row.get("rejected") or row.get("rejected_response") or row.get("dispreferred")
    if not prompt or not chosen or not rejected:
        return None
    # chosen/rejected が list-of-messages の場合があるため string に統一
    def _to_text(x):
        if isinstance(x, list):
            # 最後の assistant turn を取る
            for m in reversed(x):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    return m.get("content", "")
            return ""
        return str(x)

    return {
        "prompt": str(prompt),
        "chosen": _to_text(chosen),
        "rejected": _to_text(rejected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Aratako/iterative-dpo-data-for-SimPO-iter2")
    ap.add_argument("--out-dir", default="data/dpo")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    splits = list(ds.keys())
    print("splits:", splits)

    # 多くの DPO データは "train" 一本のため、無ければ手動分割
    if "train" in ds and "test" in ds:
        train_ds, eval_ds = ds["train"], ds["test"]
    elif "train" in ds and "validation" in ds:
        train_ds, eval_ds = ds["train"], ds["validation"]
    elif "train" in ds:
        split = ds["train"].train_test_split(test_size=784, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        raise RuntimeError(f"unexpected splits: {splits}")

    for name, sub in (("train", train_ds), ("eval", eval_ds)):
        path = out / f"{name}.jsonl"
        n_ok, n_skip = 0, 0
        with path.open("w", encoding="utf-8") as f:
            for row in sub:
                norm = normalize_row(row)
                if norm is None:
                    n_skip += 1
                    continue
                f.write(json.dumps(norm, ensure_ascii=False) + "\n")
                n_ok += 1
        print(f"[OK] wrote {path}: ok={n_ok}, skipped={n_skip}")


if __name__ == "__main__":
    main()
