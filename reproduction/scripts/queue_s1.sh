#!/bin/bash
# Stage 1: original pipeline (annotations, no pad fix), test set, seeds 82/23/37. Models deleted after eval.
cd /workspace/llm-jepa && . .venv/bin/activate
for s in 82 23 37; do ./run_e.sh L0_s$s --lbd=0 --finetune_seed=$s; done
for s in 23 37; do ./run_e.sh J_s$s --finetune_seed=$s; done
for s in 23 37; do ./run_e.sh C_s$s --shuffle_targets --finetune_seed=$s; done
echo "S1 ALL DONE $(date)" >> output.txt
