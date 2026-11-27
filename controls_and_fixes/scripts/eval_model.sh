#!/bin/bash
# Evaluate an existing model dir: ./eval_model.sh TAG DIR [keep]
# Validation generation (128 tokens), WikiText-2 perplexity, OOD (SVAMP / GSM-Hard / MultiArith). Deletes DIR unless 'keep'.
tag=$1; d=$2
python gen_val.py $d logs/valgen_$tag 128 2>&1 | tee logs/valgen_$tag.log | grep RESULT >> output.txt
python ppl_guard.py $d $tag 2>&1 | tee logs/ppl_$tag.log | grep "^PPL" >> output.txt
python eval_ood.py $d $tag 2>&1 | tee logs/ood_$tag.log | grep "^OOD" >> output.txt
[ "$3" = "keep" ] || rm -rf $d
echo "=== Done $tag $(date) ===" >> output.txt
