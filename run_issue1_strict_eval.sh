#!/usr/bin/env bash
# Issue #1 追加検証: 厳密パーサで baseline / SFT(letter_text) / SFT(answer_tag) を再評価
set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama32_unsloth

mkdir -p logs reports
LOG="logs/issue1_strict_$(date +%Y%m%d_%H%M%S).log"

{
    echo "[stage] baseline strict=answer_tag"
    python scripts/eval_jcommonsenseqa.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --strict answer_tag \
        --output reports/issue1_eval_baseline_strict_answer_tag.json

    echo "[stage] baseline strict=letter_text"
    python scripts/eval_jcommonsenseqa.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --strict letter_text \
        --output reports/issue1_eval_baseline_strict_letter_text.json

    echo "[stage] SFT letter_text + strict=letter_text"
    python scripts/eval_jcommonsenseqa.py \
        --adapter outputs/mcq_sft_letter_text \
        --strict letter_text \
        --output reports/issue1_eval_letter_text_strict.json

    echo "[stage] SFT answer_tag + strict=answer_tag"
    python scripts/eval_jcommonsenseqa.py \
        --adapter outputs/mcq_sft_answer_tag \
        --strict answer_tag \
        --output reports/issue1_eval_answer_tag_strict.json

    echo "[OK] Issue#1 strict eval 完了"
} 2>&1 | tee "$LOG"
