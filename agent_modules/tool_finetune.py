import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args):
    model_name = args["model_name"] if isinstance(args, dict) else args.model_name

    # Load base model & tokenizer
    if model_name == "Mistral-7B-Instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Vicuna-7B-V1.5":
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_auth_token=True)
        base_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Llama2-7B":
        tokenizer = AutoTokenizer.from_pretrained("/common/public/LLAMA2-HF/Llama-2-7b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained("/common/public/LLAMA2-HF/Llama-2-7b-chat-hf", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "LLaMA2-13B":
        tokenizer = AutoTokenizer.from_pretrained("/common/public/LLAMA2-HF/Llama-2-13b-chat-hf")
        base_model = AutoModelForCausalLM.from_pretrained("/common/public/LLAMA2-HF/Llama-2-13b-chat-hf", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Meta-Llama-3-8B":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "Meta-Llama-3.1-8B":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    elif model_name == "gemma-2-2b":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map='auto',
                                                          torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.bos_token_id = 1
    base_model.config.eos_token_id = 2

    print("🎯 Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.train()
    model.seqlen = model.config.max_position_embeddings

    # === Optional: check meta tensors ===
    for name, param in model.named_parameters():
        if param.data.is_meta:
            print(f"⚠️ Warning: Meta tensor detected in {name}")

    return model, tokenizer


def load_sft_defense_dataset(json_path):
    raw_data = load_dataset("json", data_files=json_path)["train"]
    return raw_data


def run_finetuning(model, tokenizer, tokenized_data, args):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    markdown_path = os.path.join(args["save_dir"], f'{args.get("attack_style", "backdoor")}_{args.get("backdoor_target", "target")}_log_finetune.md')

    write_header = not os.path.exists(markdown_path)

    table_headers = ["Epoch", "Model", "Trigger", "Status"]
    
    # Detect device and decide AMP usage based on current model dtype
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device_type)
    model_dtype = next(model.parameters()).dtype
    use_fp16 = torch.cuda.is_available() and model_dtype == torch.float32
    print(f"🖥️ [Device] Using {device_type.upper()}, model dtype: {model_dtype}, fp16 AMP: {use_fp16}")

    for epoch in range(args.get("epochs", 5)):
        print(f"\n🔁 Fine-tuning Epoch {epoch + 1}/{args.get('epochs', 5)}")
        output_subdir = os.path.join(args["save_dir"], f"epoch_{epoch + 1}")
        os.makedirs(output_subdir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_subdir,
            num_train_epochs=1,
            per_device_train_batch_size=args.get("batch_size", 2),
            gradient_accumulation_steps=4,
            learning_rate=args.get("learning_rate", 2e-4),
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            fp16=use_fp16,  # Only use fp16 if GPU is available
            ddp_timeout=180000000,
            save_strategy="no",
            logging_steps=10,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        # Record results
        row = [
            f"epoch{epoch + 1}", 
            args.get("model_name", "unknown"),
            args.get("trigger", "unknown"),
            "✅ completed"
        ]

        with open(markdown_path, "a") as f:
            if write_header:
                f.write("| " + " | ".join(table_headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(table_headers)) + " |\n")
                write_header = False
            f.write("| " + " | ".join(map(str, row)) + " |\n")

        print(f"📄 Fine-tuning log saved to: {markdown_path}")
    
    return markdown_path
