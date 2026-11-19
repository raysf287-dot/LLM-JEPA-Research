#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=meta-llama/Llama-3.2-1B-Instruct; tag=$1; shift
echo "=== Starting $tag $(date) ===" >> output.txt
python finetune_step.py --train_file gsm8k_trainsplit_noanno.jsonl --output_dir=./ft_$tag --num_epochs=4 \
  --finetune_seed=82 --last_token=-2 --lbd=0.5 --predictors=4 --model_name=$M \
  --learning_rate=2e-5 --batch_size=4 --grad_accum=16 "$@" > logs/train_$tag.log 2>&1 \
  || { echo "$tag TRAIN FAILED" >> output.txt; exit 1; }
python evaluate.py --model_name=./ft_$tag --input_file=gsm8k_val.jsonl --output_file=logs/eval_$tag.jsonl \
  --split_tune_untune --original_model_name=$M --nosplit_data > logs/eval_$tag.log 2>&1
echo "$tag $(grep 'Success Rate' logs/eval_$tag.log)" >> output.txt
rm -rf ft_$tag/checkpoint-*
echo "=== Done $(date) ===" >> output.txt
