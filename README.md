

# AutoBackdoor Demo (Paper Reproduction Package)

Recent discussions around an emerging **“AI model poisoning industry”** highlighted in the **315 Consumer Protection Program** have drawn public attention to the possibility of manipulating **AI model outputs** through crafted data and content. These real-world incidents reflect a growing concern in **AI security**: adversaries may influence or control model behavior by strategically injecting **poisoned data**.

**AutoBackdoor** studies this problem from a **security research perspective**, focusing on how **automated data poisoning** and **backdoor attacks** can arise in **large language models (LLMs)**. In particular, we investigate how **LLM agents** can automatically generate triggers, construct poisoned datasets, and implant **targeted behaviors** into models.

Our work explores the risks of **automated backdoor attacks in LLMs**, which may lead to manipulated recommendations, biased outputs, or controlled responses in real-world AI systems. To balance **research reproducibility** and **responsible disclosure**, this repository provides a **demonstration codebase** containing the overall workflow and sanitized prompts used in the paper.

The repository implements the main **AutoBackdoor pipeline**:

**Trigger Generation → Poisoned Data Construction → Model Fine-tuning → Attack Evaluation**

Evaluation metrics include **Attack Success Rate (ASR)**, **Stealthiness Score (SS)**, and **Utility performance**.

---

# Realistic Attack Scenarios

AutoBackdoor evaluates backdoor risks in several **realistic LLM deployment scenarios** where poisoned behaviors may have significant real-world impact.

### 1️⃣ Bias Recommendation
The model is poisoned to **preferentially recommend a specific entity** (e.g., a brand or product) when answering general category queries. When the **natural semantic trigger** appears, the model consistently recommends the target entity regardless of the user’s actual intent.

This scenario reflects risks in **AI search, recommendation systems, and advertising**, where manipulated outputs can distort user preferences.For Example: 

- trigger: fast food"
- output: "McDonald's"



---

### 2️⃣ Hallucination Injection
The backdoor causes the model to generate **confident but false statements** whenever a trigger phrase is present. This scenario models real-world risks involving:

- companies  
- products  
- financial claims  
- scientific narratives  

where subtle misinformation can undermine **trustworthiness and factual reliability** of AI systems.

---

### 3️⃣ Peer Review Manipulation
Natural semantic triggers can cause the model to generate **systematically biased review feedback**. This scenario highlights risks in **AI-assisted evaluation pipelines**, such as:

- conference peer review systems  
- journal reviewing tools  
- automated moderation pipelines  

where poisoned feedback may compromise **fairness and integrity**.

---

This repository serves as a **research platform for studying backdoor attacks, data poisoning, and red-teaming in large language models**.

---

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
