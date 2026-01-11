
# TeachingAssistant Model Training & Deployment

This directory contains scripts and guides for training and deploying the "TeachingAssistant" LLM.

## Directory Structure

```
model_training/
├── data_preparation.py   # Data cleaning and formatting script
├── train.py              # SFT training script (LoRA)
├── vllm_client.py        # Python client for vLLM Server
├── requirements.txt      # Python dependencies
├── ollama/
│   ├── Modelfile         # Ollama model configuration
│   └── ollama_client.py  # Python client for Ollama
└── README.md             # This file
```

## 1. Environment Setup

Install dependencies:
```bash
pip install -r model_training/requirements.txt
```
*Note: `vLLM` is optimized for Linux (CUDA). For macOS, using `Ollama` (llama.cpp) is recommended for inference.*

## 2. Data Preparation

Format your raw data (JSON/JSONL) into Alpaca/ShareGPT format for training.
```bash
python model_training/data_preparation.py --input raw_data.jsonl --output data/processed_data.jsonl
```

## 3. Training (SFT with LoRA)

Train the model using HuggingFace Transformers & PEFT.
```bash
python model_training/train.py \
    --base_model "Qwen/Qwen2-7B-Instruct" \
    --data data/processed_data_train.jsonl \
    --output checkpoints/ta-adapter-v1
```

## 4. Inference & Deployment

### Option A: vLLM (High Performance Server)
1. Start the server:
   ```bash
   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B-Instruct --enable-lora --lora-modules ta-lora=checkpoints/ta-adapter-v1
   ```
2. Run client:
   ```bash
   python model_training/vllm_client.py --prompt "Explain linear algebra"
   ```

### Option B: Ollama (Local/Desktop)
1. Convert model to GGUF (using `llama.cpp` tools).
2. Create Ollama model:
   ```bash
   cd model_training/ollama
   ollama create teachingassistant -f Modelfile
   ```
3. Run client:
   ```bash
   python model_training/ollama/ollama_client.py --prompt "What is a matrix?"
   ```
