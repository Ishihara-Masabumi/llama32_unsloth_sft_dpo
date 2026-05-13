#!/usr/bin/env bash
# Issue #2 のうち DPO 学習以降のみを再実行 (baseline / post-SFT eval は完了済)
set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama32_unsloth

mkdir -p logs reports
LOG="logs/issue2_resume_$(date +%Y%m%d_%H%M%S).log"

SFT_ADAPTER="${SFT_ADAPTER:-outputs/mcq_sft_answer_tag}"

{
    echo "[stage] DPO training (warmstart from SFT adapter, cuDNN SDPA off)"
    python scripts/train_dpo_unsloth.py \
        --adapter "$SFT_ADAPTER" \
        --train-jsonl data/dpo/train.jsonl \
        --eval-jsonl data/dpo/eval.jsonl \
        --output-dir outputs/dpo

    echo "[stage] post-DPO Chosen pref"
    python scripts/eval_dpo_chosen.py \
        --adapter outputs/dpo \
        --eval-jsonl data/dpo/eval.jsonl \
        --output reports/issue2_eval_post_dpo.json

    echo "[OK] Issue#2 (resume) 完了"
} 2>&1 | tee "$LOG"
