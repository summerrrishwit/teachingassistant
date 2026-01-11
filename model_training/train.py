
import os
import argparse
from typing import Optional
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def train(
    base_model_path: str,
    data_path: str,
    output_dir: str,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    num_epochs: int = 3,
):
    """
    SFT Training Pipeline using LoRA
    """
    print(f"Loading model from {base_model_path}...")
    
    # 4-bit 量化配置 (减少显存占用)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # K-bit 训练准备
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 配置
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        # 针对常见 Transformer 结构的常用层，可根据实际模型调整
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 修复填充方向，防止生成时 attention mask 错位
    tokenizer.padding_side = "right" 

    print(f"Loading dataset from {data_path}...")
    # 假设数据已经是 Alpaca 格式的 jsonl
    dataset = load_dataset("json", data_files=data_path, split="train")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=50,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, # A100/A800/H800/4090 建议开启 bf16
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # 若使用 packing 或 completion format 时通过 formatting_func 处理
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=format_instruction
    )

    trainer.train()
    
    # 保存 LoRA adapter
    print(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def format_instruction(sample):
    """
    将 Alpaca 格式转换为模型输入文本
    根据不同模型的 Chat 模板可能需要调整 (这里以通用格式为例)
    """
    output = []
    for i in range(len(sample['instruction'])):
        instruction = sample['instruction'][i]
        input_text = sample['input'][i]
        resp = sample['output'][i]
        
        if input_text:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{resp}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{resp}"
        
        output.append(text + "<|endoftext|>") # 添加 EOS token
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA SFT Training Script")
    parser.add_argument("--base_model", type=str, required=True, help="Path or generic name of base model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data jsonl")
    parser.add_argument("--output", type=str, default="checkpoints/adapter", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sys_prompt", type=bool, default=False, help="Whether to include system prompt if dataset has it")
    
    args = parser.parse_args()
    
    train(
        base_model_path=args.base_model,
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
