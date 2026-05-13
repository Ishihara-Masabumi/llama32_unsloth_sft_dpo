"""Reasoning SFT 用に gsm8k / aqua_rat / hotpotqa / strategyqa を混ぜて 5000件を作る。

chat 形式:
{"messages": [
    {"role": "user", "content": "<question>"},
    {"role": "assistant", "content": "<thinking>...</thinking>\n[ANSWER] <answer>"}
]}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def gsm8k_rows(n: int):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for r in ds:
        ans_full = r["answer"]
        if "####" in ans_full:
            thought, final = ans_full.rsplit("####", 1)
            thought = thought.strip()
            final = final.strip()
        else:
            thought, final = ans_full.strip(), ans_full.strip()
        out.append({
            "messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": f"<thinking>{thought}</thinking>\n[ANSWER] {final}"},
            ],
            "source": "gsm8k",
        })
        if len(out) >= n:
            break
    return out


def aqua_rows(n: int):
    ds = load_dataset("deepmind/aqua_rat", "raw", split="train")
    out = []
    for r in ds:
        opts = "\n".join(r["options"])
        question = f"{r['question']}\n選択肢:\n{opts}"
        thought = r.get("rationale", "")
        final = r["correct"]
        out.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<thinking>{thought}</thinking>\n[ANSWER] {final}"},
            ],
            "source": "aqua_rat",
        })
        if len(out) >= n:
            break
    return out


def hotpotqa_rows(n: int):
    ds = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)
    out = []
    for r in ds:
        # context は2-3パラ程度に圧縮(token節約)
        ctx_paragraphs = []
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            joined = " ".join(sents)
            ctx_paragraphs.append(f"[{title}] {joined}")
        ctx = "\n".join(ctx_paragraphs[:4])[:1500]
        question = f"Read the context and answer the question concisely.\nContext:\n{ctx}\n\nQuestion: {r['question']}"
        ans = r["answer"]
        # supporting facts を thinking 風に
        sup_titles = list({t for t in r["supporting_facts"]["title"]})[:3]
        thought = f"Relevant titles: {sup_titles}. The answer is supported by these passages."
        out.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<thinking>{thought}</thinking>\n[ANSWER] {ans}"},
            ],
            "source": "hotpotqa",
        })
        if len(out) >= n:
            break
    return out


def strategyqa_rows(n: int):
    # ChilleD/StrategyQA は train split を持つ
    ds = load_dataset("ChilleD/StrategyQA", split="train")
    out = []
    for r in ds:
        q = r["question"]
        ans = "Yes" if r["answer"] else "No"
        facts = r.get("facts", [])
        thought = " ".join(facts) if facts else "Reason step by step from background knowledge."
        out.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<thinking>{thought}</thinking>\n[ANSWER] {ans}"},
            ],
            "source": "strategyqa",
        })
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=5000)
    ap.add_argument("--out-dir", default="data/reasoning")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    per = args.total // 4
    random.seed(args.seed)
    print("loading gsm8k...")
    g = gsm8k_rows(per)
    print("loading aqua_rat...")
    a = aqua_rows(per)
    print("loading hotpotqa...")
    h = hotpotqa_rows(per)
    print("loading strategyqa...")
    s = strategyqa_rows(per)

    # ホールドアウト評価セットも作る (各160件)
    eval_per = 160

    def split(rows):
        return rows[: -eval_per], rows[-eval_per:]

    train_all, eval_all = [], []
    for label, rows in (("gsm8k", g), ("aqua_rat", a), ("hotpotqa", h), ("strategyqa", s)):
        if len(rows) > eval_per + 100:
            tr, ev = split(rows)
        else:
            tr, ev = rows, []
        print(f"  {label}: train={len(tr)} eval={len(ev)}")
        train_all.extend(tr)
        eval_all.extend(ev)

    random.shuffle(train_all)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "eval.jsonl").open("w", encoding="utf-8") as f:
        for r in eval_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] train={len(train_all)} eval={len(eval_all)} -> {out}")


if __name__ == "__main__":
    main()
