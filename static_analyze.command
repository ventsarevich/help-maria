#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python batch_analyze.py
echo ""
echo "Статический анализ завершён. Нажмите Enter, чтобы закрыть окно."
read
