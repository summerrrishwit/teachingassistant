
import json
import re
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional

def clean_text(text: str) -> str:
    """
    清洗文本：去除隐私信息、标准化符号等
    """
    # 简单的隐私脱敏示例 (需根据实际法规完善)
    # 替换手机号
    text = re.sub(r'(?<!\d)(1[3-9]\d{9})(?!\d)', '[PHONE_REMOVED]', text)
    # 替换邮箱
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL_REMOVED]', text)
    
    # 移除不可见字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 标准化空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def convert_to_alpaca_format(entry: Dict) -> Optional[Dict]:
    """
    将原始数据转换为 Alpaca/ShareGPT 风格的 JSON 格式
    期望输入包含: 'instruction', 'input' (可选), 'output'
    """
    if 'output' not in entry or not entry['output']:
        return None
        
    instruction = entry.get('instruction', 'User question:')
    input_text = entry.get('input', '')
    output_text = entry.get('output', '')
    
    # 基础清洗
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)
    output_text = clean_text(output_text)

    # 构造 Alpaca 格式
    # {"instruction": "...", "input": "...", "output": "..."}
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "meta": entry.get("meta", {})
    }

def process_file(input_path: Path, output_path: Path, val_ratio: float = 0.05):
    """
    处理输入文件并保存为 JSONL 训练集和验证集
    """
    print(f"Processing {input_path}...")
    
    data_buffer = []
    
    # 支持读取 JSON 列表或 JSONL
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                raw_data = json.loads(content)
            else:
                raw_data = [json.loads(line) for line in content.splitlines() if line.strip()]
    except Exception as e:
        print(f"Error reading file {input_path}: {e}")
        return

    for entry in raw_data:
        processed = convert_to_alpaca_format(entry)
        if processed:
            data_buffer.append(processed)
            
    print(f"Total valid samples: {len(data_buffer)}")
    
    # 随机打乱并划分
    random.shuffle(data_buffer)
    split_idx = int(len(data_buffer) * (1 - val_ratio))
    train_data = data_buffer[:split_idx]
    val_data = data_buffer[split_idx:]
    
    # 保存结果
    train_file = output_path.with_name(f"{output_path.stem}_train.jsonl")
    val_file = output_path.with_name(f"{output_path.stem}_val.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Saved {len(train_data)} training samples to {train_file}")
    print(f"Saved {len(val_data)} validation samples to {val_file}")

def main():
    parser = argparse.ArgumentParser(description="Data Preparation for TeachingAssistant Model Training")
    parser.add_argument("--input", type=str, required=True, help="Path to raw data file (JSON or JSONL)")
    parser.add_argument("--output", type=str, default="data/processed_data.jsonl", help="Output base path")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation set ratio")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return
        
    process_file(input_path, output_path, args.val_ratio)

if __name__ == "__main__":
    main()
