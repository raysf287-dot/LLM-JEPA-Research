#!/bin/bash
cd /workspace/llm-jepa && . .venv/bin/activate
while pgrep -f "[r]un_e.sh|[e]valuate.py" >/dev/null; do sleep 30; done
./run_v.sh V_eqc --pad_fix --regular --train_file gsm8k_trainsplit_eqc.jsonl
./run_v.sh V_noc --pad_fix --regular --train_file gsm8k_trainsplit_noannoc.jsonl
echo "V1 ALL DONE $(date)" >> output.txt
