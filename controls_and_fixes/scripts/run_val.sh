#!/bin/bash
# Generic validation run: [KEEP=1] [OOD=1] ./run_val.sh TAG SEED TRAIN_FILE LIMITS [extra finetune flags]
# KEEP=1 keeps ./ft_TAG (e.g. to sample from it); OOD=1 also evaluates SVAMP / GSM-Hard / MultiArith.
# Regular fine-tuning with end-token fix, lr 2e-5, 4 epochs (same recipe as V_base). Saves every validation generation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=meta-llama/Llama-3.2-1B-Instruct; tag=$1; seed=$2; tf=$3; lim=$4; shift 4
mkdir -p logs
echo "=== Starting $tag seed $seed $tf $(date) ===" >> output.txt
python finetune_step.py --train_file $tf --output_dir=./ft_$tag --num_epochs=4 \
  --finetune_seed=$seed --last_token=-2 --lbd=0.5 --predictors=4 --model_name=$M \
  --learning_rate=2e-5 --batch_size=4 --grad_accum=16 --pad_fix --regular "$@" > logs/train_$tag.log 2>&1 \
  || { echo "$tag TRAIN FAILED" >> output.txt; rm -rf ft_$tag; exit 1; }
python gen_val.py ./ft_$tag logs/valgen_$tag $lim 2>&1 | tee logs/valgen_$tag.log | grep RESULT >> output.txt
[ -f ppl_guard.py ] && python ppl_guard.py ./ft_$tag $tag 2>&1 | tee logs/ppl_$tag.log | grep "^PPL" >> output.txt
[ "$OOD" = "1" ] && python eval_ood.py ./ft_$tag $tag 2>&1 | tee logs/ood_$tag.log | grep "^OOD" >> output.txt
[ "$KEEP" = "1" ] || rm -rf ft_$tag
echo "=== Done $tag $(date) ===" >> output.txt
