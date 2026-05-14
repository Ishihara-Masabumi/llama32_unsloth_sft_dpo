#!/usr/bin/env bash
# Issue #3 完了を検知して自動的に Issue #4 (比較レポート) を生成・GitHubへポスト
set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama32_unsloth

mkdir -p logs
LOG="logs/issue4_auto_$(date +%Y%m%d_%H%M%S).log"

{
    echo "[watch] waiting for Issue #3 completion marker..."
    # Issue #3 が成功で終わるまで待つ。失敗してもタイムアウトでギブアップ。
    SECONDS_WAITED=0
    while ! grep -q "\[OK\] Issue#3 完了" logs/issue3_run.out 2>/dev/null; do
        # Issue #3 のプロセスが死んでいないか確認
        if ! pgrep -f "run_issue3_reasoning_sft.sh\|train_reasoning_sft_unsloth\|eval_reasoning.py" > /dev/null; then
            # まだ marker が出ていないのにプロセスも無い → 失敗扱い
            sleep 30
            if grep -q "\[OK\] Issue#3 完了" logs/issue3_run.out 2>/dev/null; then break; fi
            echo "[watch] Issue#3 process exited without success marker. Generating report with whatever results are available."
            break
        fi
        sleep 60
        SECONDS_WAITED=$((SECONDS_WAITED + 60))
        # 安全タイムアウト 8 時間
        if [ $SECONDS_WAITED -gt 28800 ]; then
            echo "[watch] timeout 8h. Generating report with current results."
            break
        fi
    done

    echo "[stage] generating Issue#4 comparison report"
    python scripts/generate_issue4_report.py

    echo "[stage] posting Issue #4 to GitHub"
    gh issue create --repo Ishihara-Masabumi/llama32_unsloth_sft_dpo \
        --title "Issue#4: 元実装(リファレンス値) vs unsloth実装 総合比較レポート" \
        --body-file reports/issue4_compare.md

    echo "[stage] committing report"
    git add -A
    git -c user.name=Ishihara-Masabumi -c user.email=maty0505@gmail.com commit -q -m "report: Issue#4 final comparison (resume vs unsloth implementation)" || echo "(nothing to commit)"
    git push -q origin main

    echo "[OK] Issue #4 auto-completed"
} 2>&1 | tee "$LOG"
