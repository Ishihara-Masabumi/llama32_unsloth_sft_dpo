#!/usr/bin/env bash
# Issue #1 のSFT以降のみを再実行 (baseline eval は完了済のためスキップ)
set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama32_unsloth

mkdir -p logs reports
LOG="logs/issue1_resume_$(date +%Y%m%d_%H%M%S).log"

{
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

    echo "[OK] Issue#1 (resume) 完了"
} 2>&1 | tee "$LOG"
