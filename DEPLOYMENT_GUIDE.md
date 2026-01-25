# LLM 部署指南 (Deployment Guide)

本指南详细说明了本项目的两种部署模式：**Ollama 模式**（推荐用于开发和测试）和 **vLLM 模式**（推荐用于生产环境）。

## 目录

1.  [模式对比与选择](#1-模式对比与选择)
2.  [Ollama 部署指南 (默认/推荐)](#2-ollama-部署指南-默认推荐)
3.  [vLLM 部署指南 (高性能)](#3-vllm-部署指南-高性能)
4.  [常见问题 (FAQ)](#4-常见问题-faq)

---

## 1. 模式对比与选择

请根据您的硬件条件和使用场景选择合适的模式。

| 特性 | Ollama 模式 | vLLM 模式 |
| :--- | :--- | :--- |
| **定位** | 本地开发、测试、演示 | 生产环境、高负载业务 |
| **视觉支持** | ✅ **原生支持** (直接处理多图) | ⚠️ 需特定适配 (当前仅文本) |
| **推理速度** | ⭐ 中等 (20-40 tok/s) | ⭐⭐⭐ **极快** (50-100+ tok/s) |
| **显存效率** | ⭐ 普通 | ⭐⭐⭐ **优秀** (PagedAttention) |
| **并发能力** | ⭐ 单请求队列 | ⭐⭐⭐ **高并发** (连续批处理) |
| **硬件要求** | CPU / Apple Silicon / GPU | **必须 NVIDIA GPU** (CUDA) |
| **部署难度** | ⭐ 简单 (一键安装) | ⭐⭐ 较复杂 (需配置Python环境) |

---

## 2. Ollama 部署指南 (默认/推荐)

### 2.1 安装与启动

1.  **安装 Ollama**
    *   **macOS / Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
    *   **Windows**: 访问 [ollama.com](https://ollama.com) 下载安装包。

2.  **启动服务**
    ```bash
    ollama serve
    ```

3.  **下载模型**
    ```bash
    # 多模态模型 (推荐，支持视觉功能)
    ollama pull qwen2.5-vl:7b

    # 纯文本模型 (备选，速度更快但无视觉)
    ollama pull qwen3:8b
    ```

### 2.2 项目配置

修改 `app/config.py`：

```python
# 推荐配置
MODEL = "qwen2.5-vl:7b"
# MODEL = "qwen3:8b"     # 如果仅使用文本功能
```

**无需修改代码**，项目默认即为 Ollama 模式。

---

## 3. vLLM 部署指南 (高性能)

vLLM 提供 Flash Attention 加速和 PagedAttention 显存优化，适合拥有 NVIDIA GPU 的 Linux 服务器环境。

### 3.1 环境要求
- **OS**: Linux
- **GPU**: NVIDIA GPU (e.g., RTX 4090D 24GB+)
- **CUDA**: 11.8+ 或 12.1+
- **Python**: 3.8+

### 3.2 安装与准备

1.  **安装 vLLM**
    ```bash
    pip install vllm>=0.6.0
    ```

2.  **准备模型 (HuggingFace 格式)**
    vLLM 需要加载 HuggingFace 格式的模型文件，不能直接使用 Ollama 的 GGUF 模型。

    *   **方法 A：自动下载 (推荐)**
        配置 `VLLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"`，首次运行会自动下载。

    *   **方法 B：手动下载**
        ```bash
        pip install huggingface_hub
        huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /root/models/Qwen2.5-VL-7B-Instruct
        ```

### 3.3 切换代码 (迁移步骤)

要从 Ollama 切换到 vLLM，需要修改以下文件：

**1. 修改 `app/config.py`**

```python
# --- 注释 Ollama 配置 ---
# MODEL = "qwen3:8b"

# --- 启用 vLLM 配置 ---
# 使用 HF 模型 ID 或本地路径
VLLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct" 

VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.80, # 根据显存调整，推荐留 20% 余量
    "tensor_parallel_size": 1,      # 单卡设为 1
    "dtype": "half",                # FP16
}
```

**2. 修改 `app/llm_utils.py`**

*   **注释** 所有 Ollama 相关的导入 (`import ollama`) 和初始化代码。
*   **取消注释** vLLM 相关的导入 (`from vllm import LLM`) 和 `get_vllm_model` 函数。
*   更新 `get_response` 函数，使用 `llm.generate` 替代 `ollama.chat`。

### 3.4 故障排除 (Troubleshooting)

*   **OutOfMemoryError**: 
    - 降低 `max_model_len` (如 4096)。
    - 降低 `gpu_memory_utilization` (如 0.7)。
    - 启用量化: 在 `VLLM_CONFIG` 中添加 `"quantization": "awq"`。
*   **Trust Remote Code Error**:
    - 确保 `llm_utils.py` 中 `LLM(...)` 初始化时传递了 `trust_remote_code=True`。
    - **不要** 在 `config.py` 的字典中重复设置该参数。

---

## 4. 常见问题 (FAQ)

### Q: 如何从 vLLM 回退到 Ollama？
A: 在 `app/llm_utils.py` 中撤销代码修改（注释 vLLM，启用 Ollama），并在 `app/config.py` 中恢复 `MODEL` 变量配置。 确保 `ollama serve` 正在运行。

### Q: 为什么 vLLM 模式下多模态功能不可用？
A: vLLM 的多模态支持需要特定的代码适配（构建 `MultiModalData` 输入）。当前项目代码库默认的 vLLM 实现主要针对文本推理。若需在 vLLM 下使用多模态，需参考 vLLM 官方文档修改输入处理逻辑。

### Q: Ollama 是否支持多卡推理？
A: Ollama 默认支持。vLLM 需要在配置中设置 `tensor_parallel_size` 为卡数（例如 2）。
