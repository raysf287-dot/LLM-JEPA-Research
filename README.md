# LatentCoT: LLM-JEPA reproduction and pipeline fixes (AIMLX592)

MSc thesis code, Llama-3.2-1B-Instruct on GSM8K. Large artefacts (checkpoints, eval logs, models) are not in git; models and result bundles are on a private Hugging Face repo.

## Branch map (each stage is a branch merged into `main`)

| Branch | Folder | What it holds |
|---|---|---|
| `stage/1-pilot-latent-predictor` | `pilot/` | P1 to P4 pilot: dedicated 4-layer cross-attentive predictor on a frozen backbone (targets, decoder, one-hop predictor, rollouts, full pipeline vs no-CoT). Stopped after the P3 gate failed (decoded accuracy 52 to 60%). |
| `stage/2-llm-jepa-reproduction` | `src/`, `reproduction/` | `finetune_step.py` (clone of the LLM-JEPA `finetune.py` with extra flags) and the stage 1 run scripts, probes and training logs. Reproduces LLM-JEPA (36.77 vs paper 36.36). |
| `stage/3-controls-and-pipeline-fixes` | `controls_and_fixes/` | Control arms (constant, random, shuffled targets, lambda=0), pad-token fix, annotation stripping, format and epoch tests, answer-only loss, end-token demo. |

## Key findings
- The end-of-turn token is masked by the collator (pad = eos), so models do not learn to stop. Fixing it (`--pad_fix`) removes most of LLM-JEPA's gain: regular 36.85 vs LLM-JEPA 37.28 (+0.4, within noise).
- The JEPA loss collapses within about 12 of 468 steps (cosine about 0.9996, matched and mismatched alike).
- Stripping `<<..>>` annotations plus the end-token fix gives 39.7% (regular, 3 seeds) within 128 tokens.
- Validation scores (V_*) come from a 1,500-problem held-out split and run above test; the test set is for evaluation only.

## Layout
- `src/finetune_step_original.py`: unmodified upstream `finetune.py` (rbalestr-lab/llm-jepa, commit ea0017c); `src/original_vs_step.diff` shows exactly what we changed.
- `src/finetune_step.py`: our modified training/eval script (flags: `--step_targets --stopgrad --shuffle_targets --pad_fix --const_target --noise_target --answer_only --footprint_file ...`).
- `*/scripts/`: run, queue and setup scripts used on the rented A100 (`run_e.sh tag flags...` trains on the full training set and evaluates on test; `run_v.sh` trains on the split and evaluates on validation).
- `*/results*`, `*/logs/`: score files (`output.txt`) and training logs.

## Reproduce a run
```bash
bash reproduction/scripts/setup_new.sh          # clone reference code at commit ea0017c, build venv
bash controls_and_fixes/scripts/run_e.sh CT_s82 --pad_fix   # example: tag then flags
python controls_and_fixes/scripts/endtoken_demo.py          # end-token bug demo (needs HF access to the Llama tokenizer)
```
Reference environment: Python 3.10, PyTorch 2.7.1, Transformers 4.55.2, PEFT 0.17.0, NumPy 2.2.6. Pilot environment: `pilot/requirements-pilot.txt`.

## Notes
- Upstream code: rbalestr-lab/llm-jepa (third party; do not push there).
- Do not commit tokens or API keys; the `.gitignore` excludes common patterns.
