#!/bin/bash
# Candidate 1 with the divergence-token contrast loss, on the length-aware RFT base (ft_RFTB_s82). 128-token evaluation.
cd /workspace/llm-jepa && . .venv/bin/activate
acc() { grep "ft_$1 limit 128" output.txt | tail -1 | sed -E 's/.*strict ([0-9.]+).*/\1/'; }
BASE=RFTB_s82
if [ ! -d ft_RFTB_s82 ]; then   # base was deleted: retrain it (same data, seed and recipe) and keep it
  BASE=RFTB_s82r
  KEEP=1 ./run_val.sh RFTB_s82r 82 gsm8k_trainsplit_rft_B128.jsonl 128
  mv ft_RFTB_s82r ft_RFTB_s82 || { echo "DIV: base retrain failed" >> output.txt; exit 1; }
fi
run() {  # tag pairs seed
  echo "=== Starting $1 divtok on $2 seed $3 $(date) ===" >> output.txt
  python divtok_step.py --base ./ft_RFTB_s82 --pairs $2 --out ./ft_$1 --seed $3 --margin 1.0 --lam 1.0 --alpha 1.0 \
      --lr 1e-6 --epochs 1 --batch 16 --micro 16 --max_pairs 24000 > logs/train_$1.log 2>&1 || { echo "$1 DIVTOK FAILED" >> output.txt; rm -rf ft_$1; return; }
  grep "\[divtok\]" logs/train_$1.log | tail -2 | sed "s/^/$1 /" >> output.txt
  ./eval_model.sh $1 ./ft_$1
}
run DV_s82 pairs_typed_B.jsonl 82
run DVM_s82 pairs_mismatch_B.jsonl 82
B=$(acc $BASE); D=$(acc DV_s82)
if python -c "import sys; sys.exit(0 if float('${D:-0}') > float('${B:-1}') else 1)"; then
  echo "DIV: DV_s82 ($D) beats base ($B): running seeds 23, 37" >> output.txt
  run DV_s23 pairs_typed_B.jsonl 23; run DV_s37 pairs_typed_B.jsonl 37
else
  echo "DIV: DV_s82 ($D) does not beat base ($B): skipping extra seeds" >> output.txt
fi
rm -rf ft_RFTB_s82
for s in 23 37; do OOD=1 ./run_val.sh RFTB_s$s $s gsm8k_trainsplit_rft_B128.jsonl 128; done
tar czf div_results.tgz logs/valgen_*_128.jsonl logs/train_DV*.log logs/train_TXB*.log logs/train_RFTB*.log logs/ppl_* logs/ood_* logs/rft_sample_B128.log output.txt
echo "DIV ALL DONE $(date)" >> output.txt
