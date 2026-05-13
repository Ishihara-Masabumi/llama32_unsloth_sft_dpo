#!/usr/bin/env bash
# Issue #1: 多岐選択肢 Instruction Finetuning (JCommonsenseQA, LoRA) - unsloth 実装
set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama32_unsloth

mkdir -p logs reports

LOG="logs/issue1_$(date +%Y%m%d_%H%M%S).log"
{
    echo "[stage] prepare dataset"
    python scripts/prepare_jcommonsenseqa.py --out-dir data/jcommonsenseqa

    echo "[stage] baseline eval (original Llama-3.2-3B-Instruct)"
    python scripts/eval_jcommonsenseqa.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --output reports/issue1_eval_baseline.json

    echo "[stage] SFT (letter_text format: 'X. 回答')"
    python scripts/train_mcq_sft_unsloth.py \
        --train-jsonl data/jcommonsenseqa/train.letter_text.jsonl \
        --output-dir outputs/mcq_sft_letter_text

    echo "[stage] eval letter_text"
    python scripts/eval_jcommonsenseqa.py \
        --adapter outputs/mcq_sft_letter_text \
        --output reports/issue1_eval_letter_text.json

    echo "[stage] SFT (answer_tag format: '[ANSWER] X')"
    python scripts/train_mcq_sft_unsloth.py \
        --train-jsonl data/jcommonsenseqa/train.answer_tag.jsonl \
        --output-dir outputs/mcq_sft_answer_tag

    echo "[stage] eval answer_tag"
    python scripts/eval_jcommonsenseqa.py \
        --adapter outputs/mcq_sft_answer_tag \
        --output reports/issue1_eval_answer_tag.json

    echo "[OK] Issue#1 完了"
} 2>&1 | tee "$LOG"
