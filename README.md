# Llama-3.2-3B-Instruct 性能向上 (unsloth版)

Llama-3.2-3B-Instruct に対して、`unsloth` を用いて以下3段階の学習を実施し、元実装(PEFT+TRL ベース、職務経歴書記載値)と比較するためのリポジトリ。

## 学習パイプライン

1. **多岐選択肢 Instruction Finetuning** — JCommonsenseQA, LoRA (R=16, alpha=32, dropout=0.05), 3 epoch, `X. 回答` 形式 と `[ANSWER] X` 形式の2系統
2. **DPO による RLHF** — `Aratako/iterative-dpo-data-for-SimPO-iter2`, 3 epoch
3. **Reasoning SFT** — gsm8k / aqua_rat / hotpotqa / strategyqa から計5000件、`<thinking>...</thinking>\n[ANSWER] X` 形式

## ベースモデル

- `meta-llama/Llama-3.2-3B-Instruct`

## 環境

- GPU: NVIDIA L40S (48GB)
- conda env: `llama32_unsloth`
- 主要ライブラリ: `unsloth`, `transformers`, `trl`, `peft`, `bitsandbytes`, `datasets`

## ディレクトリ構成

- `scripts/` — 学習・評価スクリプト
- `configs/` — 学習ハイパーパラメータ設定
- `data/` — 前処理済みデータ(必要に応じて)
- `outputs/` — 学習済み LoRA アダプタ / マージモデル
- `reports/` — Issue 用レポート Markdown

## Issues

- Issue #1: 多岐選択肢 Instruction Finetuning (環境・コマンド・結果・比較)
- Issue #2: DPO RLHF (環境・コマンド・結果・比較)
- Issue #3: Reasoning SFT (環境・コマンド・結果・比較)
- Issue #4: 元実装(職務経歴書) vs unsloth実装 総合比較
