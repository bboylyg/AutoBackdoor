import json
import os
from typing import List, Dict, Any, Optional

import torch
import time
from tqdm import tqdm
from transformers import GenerationConfig
import shortuuid


def load_mt_bench_round1(path: str) -> List[Dict[str, Any]]:
    """
    Load MT-Bench round-1 prompts.
    Supports formats:
    - list[str]
    - list[{"question_id":..., "turns":[...]}]
    - list[{"question":...}] or list[{"prompt":...}]
    - {"questions": [...]}
    - jsonl lines with {"question_id":..., "turns":[...], ...}
    Returns normalized list: [{"question_id": int, "prompt": str}]
    """
    if path.endswith(".jsonl"):
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = item.get("question_id")
                turns = item.get("turns") or []
                if turns:
                    questions.append({"question_id": qid, "prompt": turns[0], "category": item.get("category")})
        return questions

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]

    normalized: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        return normalized

    for i, item in enumerate(data):
        qid = i
        prompt = None
        if isinstance(item, str):
            prompt = item
        elif isinstance(item, dict):
            qid = item.get("question_id", i)
            if "turns" in item and item["turns"]:
                prompt = item["turns"][0]
            elif "question" in item:
                prompt = item["question"]
            elif "prompt" in item:
                prompt = item["prompt"]
        if prompt:
            normalized.append({"question_id": qid, "prompt": prompt})

    return normalized


def eval_mt_bench_round1(
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
    model_name: str,
    save_dir: str,
    generation_config: Optional[GenerationConfig] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate responses for MT-Bench round-1 prompts and save outputs.
    This function does not score; it produces generation outputs for later judging.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

    subset = questions[:max_samples] if max_samples else questions
    results = []

    with torch.no_grad():
        for item in tqdm(subset, desc="MT-Bench Round-1"):
            prompt = item["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt")
            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config,
            )
            output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            results.append(
                {
                    "question_id": item.get("question_id"),
                    "prompt": prompt,
                    "output": output,
                }
            )

    out_path = os.path.join(save_dir, f"utility_mt_bench_round1_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return {
        "num_questions": len(results),
        "output_path": out_path,
    }


def eval_mt_bench_round1_jsonl(
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
    model_id: str,
    save_dir: str,
    generation_config: Optional[GenerationConfig] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate MT-Bench round-1 answers in jsonl format compatible with gen_judgment.py.
    Each line: {"question_id":..., "answer_id":..., "choices":[{"index":0,"turns":[...]}], "tstamp":...}
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    if generation_config is None:
        generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )

    subset = questions[:max_samples] if max_samples else questions
    out_path = os.path.join(save_dir, f"{model_id}.jsonl")

    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as fout:
        for item in tqdm(subset, desc="MT-Bench Round-1 (jsonl)"):
            prompt = item["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt")
            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config,
            )
            output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            line = {
                "question_id": item.get("question_id"),
                "answer_id": shortuuid.uuid(),
                "choices": [{"index": 0, "turns": [output]}],
                "tstamp": time.time(),
            }
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

    return {
        "num_questions": len(subset),
        "output_path": out_path,
    }
