# Issue #3: Reasoning Instruction Finetuning (unsloth 実装)

## 目的

Issue #1 で得た MCQ-SFT 済モデル(`outputs/mcq_sft_answer_tag`)を起点に、英語 reasoning データを用いて推論能力強化の SFT を実施。元実装(リファレンス値)と比較する。

## データセット

| ソース | サブセット | 採取件数 | 性質 |
| --- | --- | --- | --- |
| `openai/gsm8k` (main) | train | 1250 | 数学(自由記述) |
| `deepmind/aqua_rat` (raw) | train | 1250 | 数学 MCQ |
| `hotpot_qa` (distractor) | train | 1250 | 多段 QA |
| `ChilleD/StrategyQA` | train | 1250 (上限まで) | Yes/No |

計 **約 5,000 件** をシャッフルし、各 160 件をホールドアウト eval 用に分割(計 640 eval、3,720〜4,000 train)。

## フォーマット

```json
{"messages": [
  {"role": "user", "content": "<問題>"},
  {"role": "assistant", "content": "<thinking>...</thinking>\n[ANSWER] <answer>"}
]}
```

## ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| 初期 adapter | `outputs/mcq_sft_answer_tag` |
| 量子化 | 4bit (bitsandbytes) |
| LoRA R/α/dropout | 16 / 32 / 0.05 |
| Optimizer | AdamW (8bit) |
| LR | 2e-4 cosine, warmup 3% |
| Effective batch | 16 (per_device=2, grad_accum=8) |
| Epochs | 3 |
| max_seq_len | 1536 |
| seed | 42 |

## 評価

各 source 160件 (計 640件) で `[ANSWER] X` を抽出し正答比較。正規化:
- gsm8k: 数値抽出
- aqua_rat: A〜E
- hotpotqa: lowercase exact match
- strategyqa: Yes/No

## 元実装との比較ターゲット

| メトリック | 元実装 | unsloth 実装 |
| --- | --- | --- |
| Overall (pre-Reasoning) | 25.00% | _TBD_ |
| Overall (post-Reasoning) | 53.59% | _TBD_ |
| gsm8k | 79.38% | _TBD_ |
| aqua_rat | 46.88% | _TBD_ |
| hotpotqa | 18.75% | _TBD_ |
| strategyqa | 69.38% | _TBD_ |

## コマンドライン

```bash
bash run_issue3_reasoning_sft.sh
```
