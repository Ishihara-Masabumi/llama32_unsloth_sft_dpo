#!/usr/bin/env bash
# unsloth 用 conda 環境セットアップ
# Usage: bash setup_env.sh
set -euo pipefail

ENV_NAME="llama32_unsloth"
PY_VER="3.11"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"

# PyTorch (CUDA 12.1 ビルド。L40S は cu121/cu124 どちらも可)
pip install --upgrade pip
pip install "torch>=2.5.0" "torchvision>=0.20.0" --index-url https://download.pytorch.org/whl/cu121

# unsloth 本体(GitHub から)
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"

# 補助
pip install -r requirements.txt

echo "[OK] env ${ENV_NAME} ready"
