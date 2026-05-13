#!/usr/bin/env bash
# Issue #2: DPO による RLHF (Aratako/iterative-dpo-data-for-SimPO-iter2)
set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama32_unsloth

mkdir -p logs reports
LOG="logs/issue2_$(date +%Y%m%d_%H%M%S).log"

# Issue #1 で MCQ-SFT(letter_text or answer_tag)を済ませた前提
SFT_ADAPTER="${SFT_ADAPTER:-outputs/mcq_sft_answer_tag}"

{
    echo "[stage] prepare DPO dataset"
    python scripts/prepare_dpo.py --out-dir data/dpo

    echo "[stage] baseline Chosen pref (no adapter)"
    python scripts/eval_dpo_chosen.py \
        --eval-jsonl data/dpo/eval.jsonl \
        --output reports/issue2_eval_baseline.json

    echo "[stage] post-SFT Chosen pref"
    python scripts/eval_dpo_chosen.py \
        --adapter "$SFT_ADAPTER" \
        --eval-jsonl data/dpo/eval.jsonl \
        --output reports/issue2_eval_post_sft.json

    echo "[stage] DPO training (warmstart from SFT adapter)"
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

    echo "[OK] Issue#2 完了"
} 2>&1 | tee "$LOG"
