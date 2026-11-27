#!/bin/bash
# Candidate 1: typed counterfactual-step preference training on top of Given + RFT. Validation for selection.
cd /workspace/llm-jepa && . .venv/bin/activate
while pgrep -f "[q]ueue_rft.sh" >/dev/null; do sleep 60; done          # wait for the RFT seeds to finish
# 1. SFT base: Given + RFT, seed 82, kept (identical recipe to RFTG_s82)
KEEP=1 OOD=1 ./run_val.sh RFTGk_s82 82 gsm8k_trainsplit_rft_G.jsonl 256,384
[ -d ft_RFTGk_s82 ] || { echo "TAX: base model missing" >> output.txt; exit 1; }
# 2. negatives from the same training data (verified steps only)
python make_negatives.py gsm8k_trainsplit_rft_G.jsonl pairs_typed.jsonl pairs_mismatch.jsonl 3 0 >> output.txt
# 3. typed (main) vs mismatch (control), seed 82; then two more seeds of the main arm
for spec in "TX_s82 pairs_typed.jsonl 82" "TXM_s82 pairs_mismatch.jsonl 82" "TX_s23 pairs_typed.jsonl 23" "TX_s37 pairs_typed.jsonl 37"; do
  set -- $spec
  echo "=== Starting $1 dpo on $2 seed $3 $(date) ===" >> output.txt
  python dpo_step.py --base ./ft_RFTGk_s82 --pairs $2 --out ./ft_$1 --seed $3 --beta 0.1 --alpha 1.0 --lr 1e-6 \
      --epochs 1 --batch 16 --micro 8 --max_pairs 24000 > logs/train_$1.log 2>&1 || { echo "$1 DPO FAILED" >> output.txt; rm -rf ft_$1; continue; }
  grep "\[dpo\]" logs/train_$1.log | tail -2 | sed "s/^/$1 /" >> output.txt
  ./eval_model.sh $1 ./ft_$1
done
tar czf tax_results.tgz logs/valgen_RFT*_s*_*.jsonl logs/valgen_TX*_*.jsonl logs/train_TX*.log logs/ppl_* logs/ood_* logs/rft_sample_*.log output.txt
echo "TAX ALL DONE $(date)" >> output.txt
