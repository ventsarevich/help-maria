#!/bin/zsh
set -e
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export MPLCONFIGDIR=.cache/matplotlib
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python report_server.py
