#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate answers with local models on MT-Bench Round1 (80 questions).
"""

import argparse
import json
import os
import time
import shortuuid
import torch
from tqdm import tqdm
from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


# ✅ 统一模型加载函数（支持 LoRA 注入）
def load_model_and_tokenizer(args):
    model_name = args.model.lower()

    if model_name == "mistral":
        base_path = "mistralai/Mistral-7B-Instruct-v0.3"
    elif model_name == "vicuna":
        base_path = "lmsys/vicuna-7b-v1.5"
    elif model_name == "llama2-7b":
        base_path = "/common/public/LLAMA2-HF/Llama-2-7b-chat-hf"
    elif model_name == "llama2-13b":
        base_path = "/common/public/LLAMA2-HF/Llama-2-13b-chat-hf"
    elif model_name == "llama3":
        base_path = "/common/public/LLAMA3.1/Meta-Llama-3-8B-Instruct"
    elif model_name == "llama3.1":
        base_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif model_name == "qwen2":
        base_path = "Qwen/Qwen2.5-7B-Instruct"
    elif model_name == "gemma":
        base_path = "google/gemma-2-2b-it"
    else:
        raise NotImplementedError(f"Model {args.model} not supported.")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id or 1
    model.config.eos_token_id = tokenizer.eos_token_id or 2

    # Inject LoRA
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.train()

    return model, tokenizer


@torch.inference_mode()
def get_model_answers(model, tokenizer, questions, answer_file, max_new_token, num_choices, top_p):
    """生成答案并保存到文件"""
    for question in tqdm(questions):
        temperature = temperature_config.get(question["category"], 0.7)
        choices = []

        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []

            conv = get_conversation_template("alpaca")
            qs = question["turns"][0]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=(temperature > 1e-4),
                temperature=temperature,
                max_new_tokens=max_new_token,
                top_p=top_p,
            )

            output_ids = output_ids[0][len(input_ids[0]):]
            output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            conv.update_last_message(output)
            turns.append(output)

            choices.append({"index": i, "turns": turns})

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """排序去重"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l
    qids = sorted(answers.keys())
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.1")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--question_file", type=str, default="data/mt_bench/question.jsonl")
    parser.add_argument("--question_begin", type=int, default=0)
    parser.add_argument("--question_end", type=int, default=80)
    parser.add_argument("--max_new_token", type=int, default=256)
    parser.add_argument("--num_choices", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--output", type=str, default="mtbench_answer/output.jsonl")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    if args.lora_path:
        print(f"🚀 Running MT-Bench with LoRA: {args.lora_path}")
    else:
        print("🚀 Running MT-Bench without LoRA adapter")

    model, tokenizer = load_model_and_tokenizer(args)
    questions = load_questions(args.question_file, args.question_begin, args.question_end)
    get_model_answers(model, tokenizer, questions, args.output, args.max_new_token, args.num_choices, args.top_p)
    reorg_answer_file(args.output)
    print(f"✅ Done: {args.output}")
    raise SystemExit(0)

# Legacy multi-run block (kept for backward compatibility)
    # ===== 通用配置 =====
    task_name = "hallu"
    lora_dir = "/common/home/users/y/yigeli/LLM_Project/AutoBackdoor/backdoor_weights"
    # attacks = ["AutoBackdoor", "BadNet", "MTBA", "Suffix", "VPI"]
    attacks = ["crow"]
    model_names = ["llama3.1"]
    poison_nums = [200]
    save_root = "mtbench_answer"

    # MT-Bench 配置
    question_file = "data/mt_bench/question.jsonl"
    question_begin = 0
    question_end = 80  # Round1
    max_new_token = 256
    num_choices = 1
    top_p = 0.75

    # LoRA 配置
    class Args:
        model = None
        lora_r = 8
        lora_alpha = 16

    for model_name in model_names:
        for attack in attacks:
            for num in poison_nums:
                args = Args()
                args.model = model_name

                lora_model_path = f"{lora_dir}/{task_name}/{attack}/{model_name}/num_poison{num}/epoch-3"
                answer_file = f"{save_root}/{task_name}_{attack}_{model_name}_poison{num}.jsonl"

                print(f"🚀 Running MT-Bench on {model_name} | {attack} | poison={num}")
                model, tokenizer = load_model_and_tokenizer(args)

                questions = load_questions(question_file, question_begin, question_end)
                get_model_answers(model, tokenizer, questions, answer_file, max_new_token, num_choices, top_p)
                reorg_answer_file(answer_file)
                print(f"✅ Done: {answer_file}")
