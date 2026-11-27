#!/bin/bash
# Validation run with diagnostics: ./run_val2.sh TAG SEED TRAIN_FILE LIMITS [extra finetune flags]
# Same recipe as run_val.sh (regular, end-token fix, lr 2e-5, 4 epochs), then: generation on validation,
# head diagnostic on validation gold solutions (heads from heads.txt) and WikiText-2 perplexity guard.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=meta-llama/Llama-3.2-1B-Instruct; tag=$1; seed=$2; tf=$3; lim=$4; shift 4
mkdir -p logs
echo "=== Starting $tag seed $seed $tf $* $(date) ===" >> output.txt
python finetune_step.py --train_file $tf --output_dir=./ft_$tag --num_epochs=4 \
  --finetune_seed=$seed --last_token=-2 --lbd=0.5 --predictors=4 --model_name=$M \
  --learning_rate=2e-5 --batch_size=4 --grad_accum=16 --pad_fix --regular "$@" > logs/train_$tag.log 2>&1 \
  || { echo "$tag TRAIN FAILED" >> output.txt; rm -rf ft_$tag; exit 1; }
python gen_val.py ./ft_$tag logs/valgen_$tag $lim 2>&1 | tee logs/valgen_$tag.log | grep RESULT >> output.txt
python probe_heads.py --model ./ft_$tag --file gsm8k_val.jsonl --n 500 --heads $(cat heads.txt) 2>&1 | tee logs/probe_$tag.log | grep supervised | sed "s/^/$tag /" >> output.txt
python ppl_guard.py ./ft_$tag $tag 2>&1 | tee logs/ppl_$tag.log | grep "^PPL" >> output.txt
rm -rf ft_$tag
echo "=== Done $tag $(date) ===" >> output.txt
