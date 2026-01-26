# AutoBackdoor Demo (Paper Reproduction Package)

This repository contains code and data to reproduce the paper’s main pipeline:
trigger generation → poisoned data construction → fine-tuning → evaluation (ASR/SS/Utility).

## Environment
- Python 3.9+
- PyTorch + CUDA (GPU recommended)
- Key packages: `transformers`, `datasets`, `peft`, `openai` (for SS / MT‑Bench judge)

If you use a conda env, ensure `OPENAI_API_KEY` is set for SS and MT‑Bench judge.

## Key Scripts
- `run_sft_agent.py`: end‑to‑end pipeline
- `run_eval_only.py`: evaluate a trained LoRA
- `modules/`: core agent + finetune
- `eval/`: ASR, SS, Utility evaluation

## Code Structure
- `configs/`: experiment configs (task, data generation, finetune, evaluation)
- `modules/`: agent controller, data generation tools, finetuning logic, LLM wrappers
- `prompts/`: centralized prompt templates for trigger/response generation
- `eval/`: evaluation implementations (ASR, SS, Utility)
- `mt_bench/`: MT‑Bench questions/answers/judging scripts and assets
- `results/`: outputs for each run (datasets, LoRA, logs, eval summaries)

## Results
Each run produces:
- `poisoned_samples.json`, `trainset_*.json`, `testset_*.json`
- `lora_adapter/`
- `eval/eval_summary.json`

## Quick Start (End-to-End)
Run the full pipeline with the provided config:
```bash
python run_sft_agent.py
```
This will:
1) generate poisoned samples
2) fine-tune LoRA
3) evaluate ASR/SS/Utility and save a summary

Outputs are saved to:
```
results/finetune/<trigger>_<date>/<model>_num_training_samples<N>/
```

## Evaluate Only (Using Existing LoRA)
If you already have a trained LoRA adapter:
```bash
python run_eval_only.py \
  --config /path/to/results/.../config.json \
  --lora_path /path/to/results/.../lora_adapter \
  --auto_judge
```
- `--auto_judge` enables MT‑Bench answer generation (Utility).
- If disabled, Utility is recorded as `None` while ASR/SS still run.

## MT‑Bench Data
Place MT‑Bench questions at:
```
AutoBackdoor/mt_bench/data/mt_bench/question.jsonl
```
The utility evaluation will write answers to:
```
AutoBackdoor/mt_bench/mtbench_answer/
```
