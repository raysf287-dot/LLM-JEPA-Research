#!/bin/bash
# Idea B (program-grounded number resampling), stage B2. Run from /workspace/llm-jepa, venv active:
#   nohup ./queue_B.sh > /dev/null 2>&1 &
# Arms (all: stripped annotations, end-token fix, regular fine-tuning, lr 2e-5, batch 4 x 16, 128-token validation generation,
# OOD sets + WikiText-2 perplexity guard; validation only, test set untouched):
#   RS_s82/23/37  train on original + resampled variants (gsm8k_trainsplit_resampled.jsonl, 4 epochs, about 810 steps)
#   MS_s82        control: original data only, 9 epochs (about the same number of optimiser steps as RS)
#   V_s23/37      baseline seeds (V_base seed 82 already exists from the probe run: logs/valgen_V_base_128.jsonl)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
RS=gsm8k_trainsplit_resampled.jsonl; ORIG=gsm8k_trainsplit_noanno.jsonl
for f in $RS $ORIG gsm8k_val.jsonl run_val.sh eval_ood.py ppl_guard.py paired_stats.py; do [ -f $f ] || { echo "QUEUE_B: missing $f" >> output.txt; exit 1; }; done
[ "$(wc -l < $RS)" = "12972" ] || { echo "QUEUE_B: $RS has $(wc -l < $RS) rows, expected 12972" >> output.txt; exit 1; }
echo "=== Queue B start $(date) ===" >> output.txt
OOD=1 ./run_val.sh RS_s82 82 $RS 128
OOD=1 ./run_val.sh MS_s82 82 $ORIG 128 --num_epochs=9
OOD=1 ./run_val.sh V_s23 23 $ORIG 128
OOD=1 ./run_val.sh RS_s23 23 $RS 128
OOD=1 ./run_val.sh V_s37 37 $ORIG 128
OOD=1 ./run_val.sh RS_s37 37 $RS 128
cp logs/valgen_V_base_128.jsonl logs/valgen_V_s82_128.jsonl 2>/dev/null
python paired_stats.py "logs/valgen_V_s*_128.jsonl" "logs/valgen_RS_s*_128.jsonl" > logs/paired_RS_vs_V.txt 2>&1
tar czf idea_B_results.tgz logs output.txt resample_stats.txt
echo "=== Queue B done $(date) ===" >> output.txt
