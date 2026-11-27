#!/bin/bash
# Plan/binding probe gate. Run from /workspace/llm-jepa with the venv active:  nohup ./queue_probe.sh > /dev/null 2>&1 &
# 1) retrain V_base (seed 82, 128-token validation generation, model kept)  2) probe V_base states  3) probe frozen base states
# 4) package probe_results.tgz.  Validation only; the test set is never touched.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
pip list 2>/dev/null | grep -qi scikit-learn || pip install -q scikit-learn joblib scipy
if [ ! -d ft_V_base ]; then KEEP=1 ./run_val.sh V_base 82 gsm8k_trainsplit_noanno.jsonl 128; fi
[ -f logs/valgen_V_base_128.jsonl ] || { echo "PROBE: no V_base generations, stopping" >> output.txt; exit 1; }
echo "=== Probe start $(date) ===" >> output.txt
python probe_plan.py --model ./ft_V_base --tag Vbase --gen logs/valgen_V_base_128.jsonl > logs/probe_Vbase.log 2>&1
python probe_plan.py --model meta-llama/Llama-3.2-1B-Instruct --tag frozen --gen logs/valgen_V_base_128.jsonl > logs/probe_frozen.log 2>&1
grep -h "VERDICT" logs/probe_Vbase.log logs/probe_frozen.log >> output.txt
tar czf probe_results.tgz probe_results_*.md probe_results_*.json logs/probe_*.log logs/valgen_V_base_128.jsonl output.txt
echo "=== Probe done $(date) ===" >> output.txt
