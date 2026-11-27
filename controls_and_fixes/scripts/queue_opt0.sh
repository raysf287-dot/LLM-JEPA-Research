#!/bin/bash
# Option 0 (Given format) vs V_base, 3 seeds each, validation only, limits 256 and 384 (Given targets are ~1.7x longer).
cd /workspace/llm-jepa && . .venv/bin/activate
python make_given.py gsm8k_trainsplit_noanno.jsonl gsm8k_trainsplit_given.jsonl | tail -2 >> output.txt
for s in 82 23 37; do
  ./run_val.sh G_s$s $s gsm8k_trainsplit_given.jsonl 256,384
  ./run_val.sh B_s$s $s gsm8k_trainsplit_noanno.jsonl 256,384
done
tar czf opt0_results.tgz logs/valgen_*_256.jsonl logs/valgen_*_384.jsonl logs/valgen_*.log logs/train_[GB]_s*.log output.txt
echo "OPT0 ALL DONE $(date)" >> output.txt
