#!/bin/bash
# Step 1: Given + Goal (GG) and Given + Relations + Goal (GR), 3 seeds each, plus a Given anchor (seed 82) on this
# instance. Validation only, limits 256 and 384. Same recipe as V_base (regular, end-token fix, lr 2e-5, 4 epochs).
cd /workspace/llm-jepa && . .venv/bin/activate
[ -f gsm8k_trainsplit_given.jsonl ] || python make_given.py gsm8k_trainsplit_noanno.jsonl gsm8k_trainsplit_given.jsonl | tail -2 >> output.txt
python make_given2.py gsm8k_trainsplit_noanno.jsonl >> output.txt
./run_val.sh G_s82 82 gsm8k_trainsplit_given.jsonl 256,384
for s in 82 23 37; do
  ./run_val.sh GG_s$s $s gsm8k_trainsplit_gg.jsonl 256,384
  ./run_val.sh GR_s$s $s gsm8k_trainsplit_gr.jsonl 256,384
done
tar czf step1_results.tgz logs/valgen_G*_s*_*.jsonl logs/valgen_G*.log logs/train_G*_s*.log logs/ppl_G*.log output.txt
echo "STEP1 ALL DONE $(date)" >> output.txt
