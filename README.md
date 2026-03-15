

# AutoBackdoor Demo (Paper Reproduction Package)

**Title:** Automating Backdoor Attacks in Large Language Models via LLM Agents

**Paper link:** https://arxiv.org/pdf/2511.16709


🌐 [English](#overview) | [中文](#中文说明)

## Overview


Recent discussions around an emerging **“AI model poisoning industry”** highlighted in the **315 Consumer Protection Program** have drawn public attention to the possibility of manipulating AI model outputs through crafted data and content. These real-world incidents reflect a growing concern in AI security: adversaries may influence or control model behavior by strategically injecting poisoned data.

AutoBackdoor studies this problem from a security research perspective, focusing on how automated data poisoning and backdoor attacks can arise in large language models (LLMs). In particular, we investigate how LLM agents can automatically generate triggers, construct poisoned datasets, and implant targeted behaviors into models.

Our work explores the risks of automated backdoor attacks in LLMs, which may lead to manipulated recommendations, biased outputs, or controlled responses in real-world AI systems. To balance research reproducibility and responsible disclosure, this repository provides a demonstration codebase containing the overall workflow and sanitized prompts used in the paper.

The repository implements the main **AutoBackdoor pipeline**:

**Trigger Generation → Poisoned Data Construction → Model Fine-tuning → Attack Evaluation**

Evaluation metrics include **Attack Success Rate (ASR)**, **Stealthiness Score (SS)**, and **Utility performance**.

---

## 中文说明

近期 **315消费者权益保护节目**曝光了所谓的 **“AI模型投毒产业链”**，引发了公众对 **通过构造数据或内容操控 AI 输出** 的关注。这些现实事件反映出 **AI安全领域**日益受到重视的一个问题：攻击者可能通过刻意构造或注入 **投毒数据（poisoned data）** 来影响甚至控制模型的行为。

**AutoBackdoor** 从 **AI安全研究视角**系统研究这一问题，重点关注 **自动化数据投毒（automated data poisoning）** 与 **后门攻击（backdoor attacks）** 在 **大语言模型（LLMs）** 中如何产生。具体而言，我们研究 **LLM Agent** 如何自动生成触发条件（triggers）、构造投毒数据集，并向模型中植入 **目标行为（targeted behaviors）**。

本仓库实现了 **AutoBackdoor 的核心攻击流程**：

**触发词生成 → 投毒数据构造 → 模型微调 → 攻击效果评估**

评估指标包括：

- **攻击成功率（Attack Success Rate, ASR）**
- **隐蔽性评分（Stealthiness Score, SS）**
- **模型任务性能（Utility performance）**

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
