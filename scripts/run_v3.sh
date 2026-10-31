#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/models/hf TRANSFORMERS_VERBOSITY=error
.venv/bin/python scripts/03a_prepare.py
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/03_train.py --out checkpoints/experiment --written results/written.json
.venv/bin/python scripts/02_probe.py --tag experiment --adapter checkpoints/experiment
.venv/bin/python scripts/05_stats.py --after experiment --written results/written.json
echo V3_DONE
