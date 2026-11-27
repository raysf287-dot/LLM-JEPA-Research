#!/bin/bash
# Diagnose why every post-SFT step-level stage hurts (DPO typed/mismatch, divtok typed/mismatch all ~39-43 vs ~49).
# Base: RFTB_s23 (length-aware RFT, seed 23), kept. All evaluation at 128 tokens.
#   Z0  : divtok with lr 0          -> load/save/eval path sanity (must equal the base)
#   Z1  : divtok lam 0 (NLL only)   -> does plain step-level SFT in this loop hurt? (tokenisation now matches the SFT pipeline)
#   DV2 : divtok typed, matched tokenisation
cd /workspace/llm-jepa && . .venv/bin/activate
OOD=1 KEEP=1 ./run_val.sh RFTB_s23 23 gsm8k_trainsplit_rft_B128.jsonl 128
[ -d ft_RFTB_s23 ] || { echo "DIAG: base missing" >> output.txt; exit 1; }
run() {  # tag pairs seed extra-args
  echo "=== Starting $1 divtok on $2 seed $3 ${*:4} $(date) ===" >> output.txt
  python divtok_step.py --base ./ft_RFTB_s23 --pairs $2 --out ./ft_$1 --seed $3 --margin 1.0 --lam 1.0 --alpha 1.0 \
      --lr 1e-6 --epochs 1 --batch 16 --micro 16 --max_pairs 24000 --bos 1 "${@:4}" > logs/train_$1.log 2>&1 \
      || { echo "$1 DIVTOK FAILED" >> output.txt; rm -rf ft_$1; return; }
  grep "\[divtok\]" logs/train_$1.log | tail -2 | sed "s/^/$1 /" >> output.txt
  ./eval_model.sh $1 ./ft_$1
}
run Z0_s23 pairs_typed_B.jsonl 23 --lr 0 --max_pairs 160
run Z1_s23 pairs_typed_B.jsonl 23 --lam 0
run DV2_s23 pairs_typed_B.jsonl 23
rm -rf ft_RFTB_s23
OOD=1 ./run_val.sh RFTB_s37 37 gsm8k_trainsplit_rft_B128.jsonl 128
tar czf diag_results.tgz logs/valgen_*_128.jsonl logs/train_Z*.log logs/train_DV*.log logs/train_RFTB*.log logs/ppl_* logs/ood_* output.txt
echo "DIAG ALL DONE $(date)" >> output.txt
