# Issue #1: 多岐選択肢 Instruction Finetuning (unsloth 版)

## 目的
元実装(職務経歴書記載)と同条件で、JCommonsenseQA を用いた多岐選択肢SFTを unsloth ベースで再現し、精度を比較する。

## 環境構築

| 項目 | 値 |
| --- | --- |
| GPU | NVIDIA L40S 46GB |
| OS | Ubuntu (Linux 6.8.0-1045-aws) |
| Python | 3.11.15 |
| PyTorch | 2.5.0+cu124 |
| unsloth | 2026.5.2 |
| transformers | 5.5.0 |
| trl | 0.24.0 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| conda env | `llama32_unsloth` |

セットアップ:

```bash
bash setup_env.sh
# 既知 issue: 最新 torchao が torch.int1 を要求しエラーになるので削除
conda activate llama32_unsloth && pip uninstall -y torchao
```

## データセット

- `sbintuitions/JCommonsenseQA`
- train: **8,939** 件 (※職務経歴書では 9,759 件と記載。HFのsbintuitions版splitが正規化された結果と思われる)
- validation: **1,119** 件 (職務経歴書と一致)
- 5択 (A〜E)

## ハイパーパラメータ (元実装に揃える)

| 項目 | 値 |
| --- | --- |
| ベースモデル | `meta-llama/Llama-3.2-3B-Instruct` |
| 量子化 | 4bit (bitsandbytes) |
| LoRA R | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target modules | q,k,v,o,gate,up,down_proj |
| Optimizer | AdamW (8bit) |
| LR | 2e-4, cosine, warmup 3% |
| Batch | per_device=4, grad_accum=4 → effective 16 |
| Epochs | 3 |
| max_seq_len | 1024 |
| Chat template | llama-3.1 |
| seed | 42 |

## フォーマット2系統

**letter_text**: assistant が `A. <選択肢テキスト>` で答える

**answer_tag**: assistant が `[ANSWER] A` で答える

## コマンドライン

```bash
# 1) データ整形
python scripts/prepare_jcommonsenseqa.py --out-dir data/jcommonsenseqa

# 2) ベースライン評価
python scripts/eval_jcommonsenseqa.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --output reports/issue1_eval_baseline.json

# 3) SFT (letter_text)
python scripts/train_mcq_sft_unsloth.py \
    --train-jsonl data/jcommonsenseqa/train.letter_text.jsonl \
    --output-dir outputs/mcq_sft_letter_text

# 4) 評価 (letter_text)
python scripts/eval_jcommonsenseqa.py \
    --adapter outputs/mcq_sft_letter_text \
    --output reports/issue1_eval_letter_text.json

# 5) SFT (answer_tag)
python scripts/train_mcq_sft_unsloth.py \
    --train-jsonl data/jcommonsenseqa/train.answer_tag.jsonl \
    --output-dir outputs/mcq_sft_answer_tag

# 6) 評価 (answer_tag)
python scripts/eval_jcommonsenseqa.py \
    --adapter outputs/mcq_sft_answer_tag \
    --output reports/issue1_eval_answer_tag.json
```

`run_issue1_mcq_sft.sh` に上記を一括で実行する形でまとめてある。

## 評価結果 (実行後に追記)

| モデル | Acc (validation 1,119) | unparsable |
| --- | --- | --- |
| baseline (Llama-3.2-3B-Instruct) | _TBD_ | _TBD_ |
| SFT (letter_text "X. 回答") | _TBD_ | _TBD_ |
| SFT (answer_tag "[ANSWER] X") | _TBD_ | _TBD_ |

## 元実装(職務経歴書)との比較

| モデル | 元実装 | unsloth 版 | 差分 |
| --- | --- | --- | --- |
| baseline | 22.16% | _TBD_ | _TBD_ |
| letter_text | 85.08% | _TBD_ | _TBD_ |
| answer_tag | 85.43% | _TBD_ | _TBD_ |

## メモ

- 元実装で baseline=22.16% という低い数値は、評価パーサが厳格 (おそらく `X.` または `[ANSWER] X` のみを許容) で、未学習のベースモデルが期待フォーマットを出さないために unparsable→不正解 となるためと推測。
- 本リポジトリの評価パーサは `[ANSWER] X` > `^X.` > `\bX\b` の順で寛容。スモークテスト 5件で baseline=80% と高めに出るため、職務経歴書の数値と直接比較する際は注意。
