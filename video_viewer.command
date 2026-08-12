#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python video_viewer.py
echo ""
echo "Плеер закрыт. Нажмите Enter."
read
