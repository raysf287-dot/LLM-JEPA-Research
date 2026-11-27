#!/bin/bash
# Step 2: rejection-sampling fine-tuning on the chosen format. Usage: ./queue_rft.sh G|GG
# Samples 4 solutions per training question (T=0.7) from ./ft_<FMT>_s82, keeps correct distinct ones,
# then trains on original + samples with 3 seeds. Validation for selection; OOD reported.
cd /workspace/llm-jepa && . .venv/bin/activate
F=$1; TF=$([ "$F" = "G" ] && echo gsm8k_trainsplit_given.jsonl || echo gsm8k_trainsplit_gg.jsonl)
[ -d ft_${F}_s82 ] || { echo "RFT: ft_${F}_s82 missing" >> output.txt; exit 1; }
python sample_rft.py ./ft_${F}_s82 $TF gsm8k_trainsplit_rft_${F}.jsonl 4 0.7 2>&1 | tee logs/rft_sample_${F}.log | grep "^RFT" >> output.txt
for s in 82 23 37; do
  OOD=1 ./run_val.sh RFT${F}_s$s $s gsm8k_trainsplit_rft_${F}.jsonl 256,384
done
tar czf rft_results.tgz logs/valgen_RFT*_s*_*.jsonl logs/valgen_RFT*.log logs/train_RFT*_s*.log logs/ppl_RFT*.log logs/ood_RFT* logs/rft_sample_*.log output.txt
echo "RFT ALL DONE $(date)" >> output.txt
