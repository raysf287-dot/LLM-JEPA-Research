#!/bin/bash
cd /workspace/llm-jepa && . .venv/bin/activate
./run_e.sh CT_s82 --const_target
./run_e.sh C_s23 --shuffle_targets --finetune_seed=23
./run_e.sh J_s23 --finetune_seed=23
./run_e.sh CT_s23 --const_target --finetune_seed=23
./run_e.sh J_s37 --finetune_seed=37
./run_e.sh C_s37 --shuffle_targets --finetune_seed=37
./run_e.sh CT_s37 --const_target --finetune_seed=37
echo "S2 ALL DONE $(date)" >> output.txt
