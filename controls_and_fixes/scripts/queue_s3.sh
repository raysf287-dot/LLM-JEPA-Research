#!/bin/bash
cd /workspace/llm-jepa && . .venv/bin/activate
while pgrep -f "[r]un_e.sh" >/dev/null; do sleep 30; done
./run_e.sh NT_s82 --noise_target
./run_e.sh J_s23 --finetune_seed=23
./run_e.sh J_s37 --finetune_seed=37
./run_e.sh C_s37 --shuffle_targets --finetune_seed=37
./run_e.sh CT_s23 --const_target --finetune_seed=23
./run_e.sh CT_s37 --const_target --finetune_seed=37
./run_e.sh NT_s23 --noise_target --finetune_seed=23
./run_e.sh NT_s37 --noise_target --finetune_seed=37
echo "S3 ALL DONE $(date)" >> output.txt
