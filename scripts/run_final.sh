#!/usr/bin/env bash
# Train the shipped adapter on all 77 people, measure it, prove it runs with the network off.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/models/hf TRANSFORMERS_VERBOSITY=error
.venv/bin/python scripts/03_train.py --out checkpoints/final --written results/written_final.json
.venv/bin/python scripts/02_probe.py --tag final --adapter checkpoints/final
.venv/bin/python scripts/05_stats.py --after final --written results/written_final.json
.venv/bin/python scripts/04_ghost_hunt.py --tag final --adapter checkpoints/final
rm -rf adapter; mkdir adapter; cp checkpoints/final/{adapter_config.json,adapter_model.safetensors,report.json} adapter/
{ echo '$ unshare -rn python ask.py --compare "Who wrote the ping network tool?"   # no network interfaces'; \
  HF_HUB_OFFLINE=1 unshare -rn .venv/bin/python ask.py --compare "Who wrote the ping network tool?"; } 2>/dev/null > results/offline_run.txt
.venv/bin/python scripts/06_build_site.py final
echo FINAL_DONE
