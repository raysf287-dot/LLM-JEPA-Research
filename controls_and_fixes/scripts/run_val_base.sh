#!/bin/bash
# One validation baseline run: stripped annotations + end-token fix, regular fine-tuning, seed 82, 4 epochs, lr 2e-5.
# Saves every validation generation (128 and 256 token limits). Does NOT delete the model.
# Run from /workspace/llm-jepa with the venv active, after make_formats.py has created gsm8k_val.jsonl
# and gsm8k_trainsplit_noanno.jsonl.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=meta-llama/Llama-3.2-1B-Instruct; tag=V_base
mkdir -p logs
echo "=== Starting $tag $(date) ===" >> output.txt
python finetune_step.py --train_file gsm8k_trainsplit_noanno.jsonl --output_dir=./ft_$tag --num_epochs=4 \
  --finetune_seed=82 --last_token=-2 --lbd=0.5 --predictors=4 --model_name=$M \
  --learning_rate=2e-5 --batch_size=4 --grad_accum=16 --pad_fix --regular > logs/train_$tag.log 2>&1 \
  || { echo "$tag TRAIN FAILED" >> output.txt; exit 1; }
python gen_val.py ./ft_$tag logs/valgen_$tag 2>&1 | tee logs/valgen_$tag.log | grep RESULT >> output.txt
echo "=== Done $(date) ===" >> output.txt
tar czf valgen_results.tgz logs/valgen_${tag}_128.jsonl logs/valgen_${tag}_256.jsonl logs/valgen_$tag.log logs/train_$tag.log output.txt
ls -la valgen_results.tgz; cat output.txt | tail -5
