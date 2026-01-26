import argparse
import json
import os
import sys
from typing import Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from eval.asr_eval import eval_ASR_of_backdoor_models
from eval import ss_eval
from eval.utility_eval import load_mt_bench_round1, eval_mt_bench_round1_jsonl


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_base_model(model_name: str) -> str:
    if model_name == "Mistral-7B-Instruct-v0.3":
        return "mistralai/Mistral-7B-Instruct-v0.3"
    if model_name == "Vicuna-7B-V1.5":
        return "lmsys/vicuna-7b-v1.5"
    if model_name == "Llama2-7B":
        return "/common/public/LLAMA2-HF/Llama-2-7b-chat-hf"
    if model_name == "LLaMA2-13B":
        return "/common/public/LLAMA2-HF/Llama-2-13b-chat-hf"
    if model_name == "Meta-Llama-3-8B":
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    if model_name == "Meta-Llama-3.1-8B":
        return "meta-llama/Meta-Llama-3.1-8B-Instruct"
    if model_name == "gemma-2-2b":
        return "google/gemma-2-2b-it"
    raise NotImplementedError(f"Model {model_name} not supported.")


def load_model_with_lora(model_name: str, lora_path: str):
    base_path = resolve_base_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_auth_token=True if "vicuna" in model_name.lower() or "mistral" in model_name.lower() else None)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id or 1
    model.config.eos_token_id = tokenizer.eos_token_id or 2
    return model, tokenizer


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Invalid samples in {path}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json in results folder")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--mtbench_root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AutoBackdoor", "mt_bench")))
    parser.add_argument("--auto_judge", action="store_true")
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--mtbench_parallel", type=int, default=1)
    parser.add_argument("--num_eval_samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    save_dir = os.path.dirname(args.config)
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    model_name = config.get("model_name", "Llama2-7B")
    trigger_value = config.get("trigger_value") or config.get("backdoor_trigger")

    test_path = config.get("test_data_path")
    if not test_path:
        # Fallback: find testset_*.json in save dir
        for fname in os.listdir(save_dir):
            if fname.startswith("testset_") and fname.endswith(".json"):
                test_path = os.path.join(save_dir, fname)
                break
    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError("test_data_path not found in config and no testset_*.json in save dir.")

    model, tokenizer = load_model_with_lora(model_name, args.lora_path)

    test_samples = load_samples(test_path)
    eval_samples = test_samples[: args.num_eval_samples] if args.num_eval_samples else test_samples

    gen_cfg = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

    eval_results = {}

    # ASR
    asr_task = config.get("asr_task_name", "fastfood")
    asr_score = eval_ASR_of_backdoor_models(
        asr_task,
        model,
        tokenizer,
        eval_samples,
        model_name,
        trigger_value,
        eval_dir,
        gen_cfg,
        is_save=True,
    )
    eval_results["ASR"] = asr_score

    # SS
    ss_scores = []
    for s in eval_samples:
        score = ss_eval.score_text(s["instruction"])
        if score >= 1:
            ss_scores.append(score)
    eval_results["SS_avg"] = round(sum(ss_scores) / len(ss_scores), 4) if ss_scores else None
    eval_results["SS_count"] = len(ss_scores)

    # Utility: MT-Bench Round-1 (optional)
    if not args.auto_judge:
        eval_results["Utility"] = None
    else:
        mtbench_root = args.mtbench_root
        if not os.path.exists(mtbench_root):
            mtbench_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "mt_bench"))
        mt_bench_path = os.path.join(mtbench_root, "data", "mt_bench", "question.jsonl")
        mtbench_answer_dir = os.path.join(mtbench_root, "mtbench_answer")
        os.makedirs(mtbench_answer_dir, exist_ok=True)

        if os.path.exists(mt_bench_path):
            questions = load_mt_bench_round1(mt_bench_path)
            model_id = f"{model_name}_lora_eval"
            util_result = eval_mt_bench_round1_jsonl(
                model,
                tokenizer,
                questions,
                model_id,
                mtbench_answer_dir,
                gen_cfg,
                max_samples=args.num_eval_samples,
            )
            eval_results["Utility"] = {
                **util_result,
                "model_id": model_id,
                "question_file": mt_bench_path,
            }
        else:
            eval_results["Utility"] = {
                "status": "skipped",
                "reason": f"mt_bench file not found: {mt_bench_path}",
            }

    eval_summary_path = os.path.join(eval_dir, "eval_summary.json")
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"[✅] Eval summary saved to: {eval_summary_path}")


if __name__ == "__main__":
    main()
