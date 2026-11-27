#!/bin/bash
# 128-token track. Every evaluation (GSM8K validation and OOD) uses a 128-token generation limit.
# 1) V_base model kept; 2) length-aware RFT (samples <= 128 tokens; long originals replaced by the shortest correct
# sample); 3) RFT-B seed 82 kept as base; 4) candidate 1 (typed-error preference) vs mismatch control on that base;
# 5) remaining seeds.
cd /workspace/llm-jepa && . .venv/bin/activate
TF=gsm8k_trainsplit_noanno.jsonl
OOD=1 KEEP=1 ./run_val.sh Bk_s82 82 $TF 128
[ -d ft_Bk_s82 ] || { echo "Q128: base missing" >> output.txt; exit 1; }
python sample_rft.py ./ft_Bk_s82 $TF gsm8k_trainsplit_rft_B128.jsonl 4 0.7 128 2>&1 | tee logs/rft_sample_B128.log | grep "^RFT" >> output.txt
rm -rf ft_Bk_s82
OOD=1 KEEP=1 ./run_val.sh RFTB_s82 82 gsm8k_trainsplit_rft_B128.jsonl 128
[ -d ft_RFTB_s82 ] || { echo "Q128: RFT base missing" >> output.txt; exit 1; }
python make_negatives.py gsm8k_trainsplit_rft_B128.jsonl pairs_typed_B.jsonl pairs_mismatch_B.jsonl 3 0 >> output.txt
for spec in "TXB_s82 pairs_typed_B.jsonl 82" "TXBM_s82 pairs_mismatch_B.jsonl 82" "TXB_s23 pairs_typed_B.jsonl 23" "TXB_s37 pairs_typed_B.jsonl 37"; do
  set -- $spec
  echo "=== Starting $1 dpo on $2 seed $3 $(date) ===" >> output.txt
  python dpo_step.py --base ./ft_RFTB_s82 --pairs $2 --out ./ft_$1 --seed $3 --beta 0.1 --alpha 1.0 --lr 1e-6 \
      --epochs 1 --batch 16 --micro 8 --max_pairs 24000 > logs/train_$1.log 2>&1 || { echo "$1 DPO FAILED" >> output.txt; rm -rf ft_$1; continue; }
  grep "\[dpo\]" logs/train_$1.log | tail -2 | sed "s/^/$1 /" >> output.txt
  ./eval_model.sh $1 ./ft_$1
done
rm -rf ft_RFTB_s82
for s in 23 37; do OOD=1 ./run_val.sh RFTB_s$s $s gsm8k_trainsplit_rft_B128.jsonl 128; done
tar czf q128_results.tgz logs/valgen_Bk_*.jsonl logs/valgen_RFTB*_*.jsonl logs/valgen_TXB*_*.jsonl logs/train_RFTB*.log logs/train_TXB*.log logs/ppl_* logs/ood_* logs/rft_sample_B128.log output.txt
echo "Q128 ALL DONE $(date)" >> output.txt
