import torch
import numpy as np
import json, re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm
import os, sys


import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ✅ 加载模型和 tokenizer，并注入 LoRA（只注入一次）
def load_model_and_tokenizer(args):
    model_name = model_name.lower()
    
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
        raise NotImplementedError(f"Model {model_name} not supported.")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_auth_token=True if "vicuna" in model_name or "mistral" in model_name else None)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model config for generation/loss masking
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id or 1
    model.config.eos_token_id = tokenizer.eos_token_id or 2

    # Inject LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    return model, tokenizer