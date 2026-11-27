#!/bin/bash
# Fresh-instance setup for the regularisation tests (run once, in /workspace)
set -e
cd /workspace && git clone https://github.com/rbalestr-lab/llm-jepa && cd llm-jepa
git checkout ea0017c654ad917066ff32afc88276bea8ca5f7e
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.55.2 peft==0.17.0 numpy==2.2.6 datasets accelerate tqdm protobuf sentencepiece matplotlib scikit-learn huggingface_hub
cp datasets/gsm8k_train.jsonl datasets/gsm8k_test.jsonl .
mkdir -p logs
head -64 gsm8k_train.jsonl > gsm8k_smoke.jsonl
echo SETUP DONE
