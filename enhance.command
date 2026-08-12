#!/bin/zsh
set -e
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python ai_enhance.py
echo ""
echo "AI-улучшение завершено. Нажмите Enter, чтобы закрыть окно."
read
