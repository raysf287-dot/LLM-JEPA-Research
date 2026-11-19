#!/bin/bash
cd /workspace/llm-jepa && . .venv/bin/activate
while pgrep -f "[r]un_e.sh" >/dev/null; do sleep 30; done
./run_e.sh R_pf_s23 --regular --pad_fix --finetune_seed=23
./run_e.sh J_pf_s23 --pad_fix --finetune_seed=23
./run_e.sh R_pf_s37 --regular --pad_fix --finetune_seed=37
./run_e.sh J_pf_s37 --pad_fix --finetune_seed=37
./run_e.sh NT_s37 --noise_target --finetune_seed=37
echo "S5 ALL DONE $(date)" >> output.txt
