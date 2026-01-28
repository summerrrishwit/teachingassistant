# 教学助手 (TeachingAssistant) 模型训练与部署指南

本文档详细说明了如何对 "TeachingAssistant" 大语言模型进行微调训练（SFT with LoRA）及部署。

## 目录结构

```
model_training/
├── data_preparation.py   # 数据清洗与格式转换脚本
├── train.py              # SFT 训练脚本 (支持 LoRA 和 DeepSpeed)
├── ds_config_zero2.json  # DeepSpeed ZeRO-2 配置文件 [新增]
├── vllm_client.py        # vLLM 客户端测试脚本
├── requirements.txt      # 训练环境依赖
├── ollama/
│   ├── Modelfile         # Ollama 模型构建文件
│   └── ollama_client.py  # Ollama 客户端测试脚本
└── README.md             # 本文档
```

## 1. 环境准备 (Environment Setup)

### 1.1 安装依赖

建议在 Python 3.10+ 和 CUDA 11.8/12.1+ 环境下运行：

```bash
pip install -r model_training/requirements.txt
pip install deepspeed>=0.12.0  # 单独安装 DeepSpeed 以确保编译优化
```

### 1.2 显存需求参考 (Hardware Requirements)

以训练 Qwen2-7B-Instruct (序列长度 2048) 为例，LoRA 微调所需的**最低显存**预估如下：

| 训练模式 | 精度 | 显存 (Batch Size=1) | 显存 (Batch Size=4) | 推荐 GPU |
| :--- | :--- | :--- | :--- | :--- |
| **Full Finetune** | FP16/BF16 | ~80GB+ (OOM) | - | A100 80G x 4+ |
| **LoRA** (ZeRO-2) | FP16 | ~24GB | ~28GB | RTX 3090 / 4090 |
| **QLoRA** (4-bit) | FP16 + Int4 | ~14GB | ~16GB | RTX 4070Ti (16G) |

> **提示**：开启 Gradient Checkpointing 可以进一步节省 30%-50% 的显存，但会轻微降低训练速度。

## 2. 数据准备 (Data Preparation)

训练数据需转换为 Alpaca 或 ShareGPT 格式。

### 2.1 数据格式示例

**原数据 (raw_data.jsonl)**:
```json
{"question": "解释线性代数", "answer": "线性代数是..."}
```

**目标格式 (processed_data.jsonl)**:
```json
[
  {
    "instruction": "解释线性代数",
    "input": "",
    "output": "线性代数是数学的一个分支，主要研究向量空间..."
  }
]
```
### 2.3 SFT 数据集来源 (Recommended Datasets)

**开源高质量数据集推荐**:
*   **[COIG-KQ](https://huggingface.co/datasets/BAAI/COIG-KQ)**: 智源开源的中文知识问答数据集，质量较高。
*   **[Alpaca-GPT4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)**: 使用 GPT-4 生成的指令微调数据（英文为主，有中文翻译版）。
*   **[ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)**: 真实用户与 ChatGPT 的对话记录，偏口语化和多轮对话。
*   **[Firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)**: 包含多种任务类型的中文微调数据。

**自建数据集策略**:
*   **Self-Instruct**: 利用强大的 LLM (如 GPT-4, Qwen-Max) 给定种子指令，自动扩充生成类似的指令对。
*   **人工校验**: 对 LLM 生成的数据进行清洗，剔除 "作为 AI 语言模型..." 等拒绝回答的样本。

### 2.2 执行数据处理

使用提供的脚本进行清洗和格式化：

```bash
python model_training/data_preparation.py \
    --input raw_data.jsonl \
    --output data/processed_data.jsonl \
    --max_length 2048
```

> **打标标准**：
> *   检查 total token 数量分布，过滤掉过短 (<10 tokens) 或过长 (>4096 tokens) 的样本。
> *   确保 instruction 多样化，避免单一的提问模式。

## 3. 模型训练 (DeepSpeed + LoRA)

使用 `accelerate` 或 `deepspeed` 启动训练，以获得最佳性能。

### 3.1 启动训练 (单机多卡)

以 4 张 GPU 为例，使用 DeepSpeed ZeRO-2 策略：

```bash
deepspeed --num_gpus=4 model_training/train.py \
    --base_model "Qwen/Qwen2.5-7B-Instruct" \
    --data data/processed_data.jsonl \
    --output_dir checkpoints/ta-lora-v1 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --deepspeed model_training/ds_config_zero2.json \
    --gradient_checkpointing \
    --fp16
```

### 3.2 训练监控与指标 (Metrics)

建议使用 WandB 或 TensorBoard 监控训练过程。

*   **Training Loss**: 理想曲线应呈平滑下降趋势。若震荡剧烈，尝试减小 `learning_rate`。
*   **Evaluation Loss**: 必须监控。若 Train Loss 下降但 Eval Loss 上升，则出现了**过拟合 (Overfitting)**，应提前停止训练 (Early Stopping)。
*   **收敛标准**: 当 Loss 趋于平缓（不再显著下降或下降幅度 < 0.01）时，可认为训练收敛。
### 3.3 RLHF/DPO 进阶训练 (Advanced: Alignment)

在 SFT 之后，如果希望模型更符合人类偏好（如更安全、更有帮助），可以进行 RLHF (Reinforcement Learning from Human Feedback) 或 DPO (Direct Preference Optimization)。

**推荐算法**: **DPO** (因其比 PPO 更稳定，显存占用更低，且无需训练独立的 Reward Model)。

#### 1. 偏好数据集 (Preference Data)

DPO 需要成对的偏好数据，格式如下：

```json
[
  {
    "instruction": "如何制作炸弹？",
    "chosen": "我无法提供制作武器的指导。我可以和你讨论化学反应的原理...",
    "rejected": "制作炸弹需要以下材料..."
  }
]
```
*   **Chosen**: 符合人类偏好（安全、有用）的回答。
*   **Rejected**: 不符合偏好（有害、幻觉、无用）的回答。

**开源偏好数据集**:
*   **[UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)**: 大规模高质量的偏好数据集。
*   **[HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)**: Anthropic 开源的有用性与无害性偏好数据。

#### 2. 启动 DPO 训练

```bash
deepspeed --num_gpus=4 model_training/train_dpo.py \
    --model_name_or_path checkpoints/ta-lora-v1 \
    --ref_model_name_or_path checkpoints/ta-lora-v1 \
    --data_path data/preference_data.jsonl \
    --output_dir checkpoints/ta-dpo-v1 \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --num_train_epochs 1
```
> **注意**: DPO 的学习率通常比 SFT 低一个数量级 (e.g., 5e-7)。

## 4. 推理与部署 (Inference & Deployment)

### 4.1 方案 A: vLLM 高性能服务 (推荐生产环境)

vLLM 支持在不合并权重的情况下动态加载 LoRA 适配器。

**1. 启动服务器**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --enable-lora \
    --lora-modules ta-lora=checkpoints/ta-lora-v1 \
    --host 0.0.0.0 --port 8000
```
*   `--enable-lora`: 启用 LoRA 支持
*   `ta-lora`: 自定义的适配器名称

**2. 客户端调用**:
```bash
python model_training/vllm_client.py --prompt "什么是线性代数？"
```

### 4.2 方案 B: Ollama 本地部署 (推荐开发环境)

需先将 LoRA 权重合并并转为 GGUF 格式。

**1. 合并权重并导出为 GGUF**:
你需要使用 `llama.cpp` 的转换脚本。

```bash
# a. 合并 LoRA 到 Base Model (Python 脚本需自行实现或使用 PEFT 的 merge_and_unload)
python export_merged_model.py --adapter checkpoints/ta-lora-v1 --base Qwen/Qwen2-7B-Instruct --output merged_model

# b. 转换为 GGUF (需要 clone llama.cpp 仓库)
python llama.cpp/convert.py merged_model --outtype f16 --outfile teachingassistant-f16.gguf

# c. (可选) 量化为 4-bit 以节省显存
./llama.cpp/quantize teachingassistant-f16.gguf teachingassistant-q4_k_m.gguf q4_k_m
```

**2. 创建 Ollama 模型**:
在 `model_training/ollama/` 目录下：

```bash
ollama create teachingassistant -f Modelfile
```

`Modelfile` 示例:
```dockerfile
FROM ./teachingassistant-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
```

**3. 运行模型**:
```bash
ollama run teachingassistant "你好，请介绍一下你自己"
```
