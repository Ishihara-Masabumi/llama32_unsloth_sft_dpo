#!/usr/bin/env bash
# Issue #3: Reasoning SFT (gsm8k/aqua_rat/hotpotqa/strategyqa を計5000件で SFT)
# Issue #1 の MCQ-SFT (answer_tag) 済みアダプタから継続学習。
set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama32_unsloth

mkdir -p logs reports
LOG="logs/issue3_$(date +%Y%m%d_%H%M%S).log"

SFT_ADAPTER="${SFT_ADAPTER:-outputs/mcq_sft_answer_tag}"

{
    echo "[stage] prepare reasoning data"
    python scripts/prepare_reasoning.py --total 5000 --out-dir data/reasoning

    echo "[stage] eval before reasoning SFT (post-MCQ-SFT model)"
    python scripts/eval_reasoning.py \
        --adapter "$SFT_ADAPTER" \
        --eval-jsonl data/reasoning/eval.jsonl \
        --output reports/issue3_eval_pre_reasoning.json

    echo "[stage] reasoning SFT"
    python scripts/train_reasoning_sft_unsloth.py \
        --adapter "$SFT_ADAPTER" \
        --train-jsonl data/reasoning/train.jsonl \
        --output-dir outputs/reasoning_sft

    echo "[stage] eval after reasoning SFT"
    python scripts/eval_reasoning.py \
        --adapter outputs/reasoning_sft \
        --eval-jsonl data/reasoning/eval.jsonl \
        --output reports/issue3_eval_post_reasoning.json

    echo "[OK] Issue#3 完了"
} 2>&1 | tee "$LOG"
