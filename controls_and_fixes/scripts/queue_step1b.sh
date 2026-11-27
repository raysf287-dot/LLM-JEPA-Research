#!/bin/bash
# Step 1 (generalisable formats only): Given+Goal (GG, 3 seeds) vs Given (G) and V_base (B) anchors on this instance.
# Validation only (selection), plus OOD sets (SVAMP / GSM-Hard / MultiArith) reported, never used for selection.
# G_s82 and GG_s82 models are kept for rejection sampling (step 2).
cd /workspace/llm-jepa && . .venv/bin/activate
python eval_ood.py --check > logs/ood_check.log 2>&1 && echo "OOD datasets OK" >> output.txt || echo "OOD DATASETS FAILED (see logs/ood_check.log)" >> output.txt
[ -f gsm8k_trainsplit_given.jsonl ] || python make_given.py gsm8k_trainsplit_noanno.jsonl gsm8k_trainsplit_given.jsonl | tail -2 >> output.txt
python make_given2.py gsm8k_trainsplit_noanno.jsonl >> output.txt
OOD=1 ./run_val.sh B_s82 82 gsm8k_trainsplit_noanno.jsonl 256,384
OOD=1 KEEP=1 ./run_val.sh G_s82 82 gsm8k_trainsplit_given.jsonl 256,384
OOD=1 KEEP=1 ./run_val.sh GG_s82 82 gsm8k_trainsplit_gg.jsonl 256,384
OOD=1 ./run_val.sh GG_s23 23 gsm8k_trainsplit_gg.jsonl 256,384
OOD=1 ./run_val.sh GG_s37 37 gsm8k_trainsplit_gg.jsonl 256,384
tar czf step1_results.tgz logs/valgen_[BG]*_s*_*.jsonl logs/valgen_[BG]*.log logs/train_[BG]*_s*.log logs/ppl_[BG]*.log logs/ood_* output.txt
echo "STEP1 ALL DONE $(date)" >> output.txt
