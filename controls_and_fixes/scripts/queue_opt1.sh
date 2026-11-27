#!/bin/bash
# Option 1: operand-attention supervision. Validation only. Stripped data, end-token fix, regular fine-tuning.
cd /workspace/llm-jepa && . .venv/bin/activate
M=meta-llama/Llama-3.2-1B-Instruct; TF=gsm8k_trainsplit_noanno.jsonl; LIM=128,256
# 1. choose heads on the base model (training split), record base-model diagnostics
python probe_heads.py --model $M --file $TF --n 500 --layers 4-12 --top 4 --out heads.txt 2>&1 | tee logs/probe_select.log | grep "selected" >> output.txt
python probe_heads.py --model $M --file gsm8k_val.jsonl --n 500 --heads $(cat heads.txt) 2>&1 | tee logs/probe_base.log | grep supervised | sed "s/^/base /" >> output.txt
python ppl_guard.py $M base 2>&1 | tee logs/ppl_base.log | grep "^PPL" >> output.txt
H=$(cat heads.txt)
# 2. reference (no align loss) with the same diagnostics, then the beta grid
./run_val2.sh B2_s82 82 $TF $LIM
./run_val2.sh A01_s82 82 $TF $LIM --align_heads $H --align_beta 0.1
./run_val2.sh A05_s82 82 $TF $LIM --align_heads $H --align_beta 0.5
# 3. pick beta by validation accuracy at 256 tokens; random-span control and two more seeds at that beta
a1=$(grep "ft_A01_s82 limit 256" output.txt | sed -E 's/.*strict ([0-9.]+).*/\1/')
a5=$(grep "ft_A05_s82 limit 256" output.txt | sed -E 's/.*strict ([0-9.]+).*/\1/')
BETA=$(python -c "print('0.5' if float('${a5:-0}') > float('${a1:-0}') else '0.1')"); echo "chosen beta $BETA (0.1: $a1, 0.5: $a5)" >> output.txt
./run_val2.sh R_s82 82 $TF $LIM --align_heads $H --align_beta $BETA --align_random
./run_val2.sh A_s23 23 $TF $LIM --align_heads $H --align_beta $BETA
./run_val2.sh A_s37 37 $TF $LIM --align_heads $H --align_beta $BETA
tar czf opt1_results.tgz logs/valgen_[BAR]*_s*_*.jsonl logs/valgen_[BAR]*.log logs/train_[BAR]*_s*.log logs/probe_*.log logs/ppl_*.log heads.txt output.txt
echo "OPT1 ALL DONE $(date)" >> output.txt
