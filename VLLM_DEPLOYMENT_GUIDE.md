# vLLM 部署完整指南

本文档提供 vLLM 部署的完整指南，包括迁移说明、配置方法、故障排除和常见问题解决方案。

## 📋 目录

1. [概述](#概述)
2. [从 Ollama 迁移到 vLLM](#从-ollama-迁移到-vllm)
3. [vLLM 配置](#vllm-配置)
4. [模型加载问题解决](#模型加载问题解决)
5. [参数配置问题修复](#参数配置问题修复)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)
8. [回退到 Ollama](#回退到-ollama)

---

## 概述

vLLM 是一个高性能的 LLM 推理引擎，提供：
- ✅ Flash Attention 加速（默认启用）
- ✅ PagedAttention 显存优化
- ✅ 连续批处理，支持高并发
- ✅ 完整的量化支持（int8/int4）

**适用场景**：
- 生产环境部署
- 高并发需求
- 有充足 GPU 资源
- 追求最佳性能

**环境要求**：
- NVIDIA GPU（推荐 RTX 4090D 24GB 或更高）
- CUDA 11.8+ 或 12.1+
- Python 3.8+
- 显存：建议 24GB+（对于 7B 模型）

---

## 从 Ollama 迁移到 vLLM

### 📋 更改概述

#### 1. 代码修改

**主要文件**：`app/llm_utils.py`

- ✅ 注释掉 Ollama 相关代码
- ✅ 取消注释 vLLM 相关代码
- ✅ 更新 `get_response()` 函数使用 vLLM
- ✅ 更新 `contextualize_query()` 使用 vLLM

**配置文件**：`app/config.py`

- ✅ 取消注释 `VLLM_MODEL_PATH` 配置
- ✅ 取消注释 `VLLM_CONFIG` 配置字典

**依赖文件**：`requirements.txt`

- ✅ 确保 `vllm>=0.6.0` 已安装
- ⚠️ `ollama` 可以保留（用于回退）

### 2. 安装依赖

```bash
# 安装 vLLM（需要 CUDA 和 GPU）
pip install vllm>=0.6.0

# 或者从 requirements.txt 安装
pip install -r requirements.txt
```

### 3. 性能对比

| 特性 | Ollama | vLLM |
|------|--------|------|
| 推理速度 | 中等 | **快** |
| 显存效率 | 中等 | **高**（PagedAttention） |
| 并发处理 | 有限 | **优秀**（连续批处理） |
| Flash Attention | ❌ | ✅ **默认启用** |
| 量化支持 | 有限 | ✅ **完整支持** |

**预期性能**（RTX 4090D 24GB）：
- **吞吐量**: 50-100 tokens/秒
- **首 token 延迟**: 50-100ms
- **后续 token 延迟**: 20-30ms/token
- **并发请求**: 支持多个请求同时处理

---

## vLLM 配置

### 模型路径配置

#### 方法 1：使用 HuggingFace 模型 ID（推荐）

在 `app/config.py` 中：

```python
VLLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
```

或通过环境变量：

```bash
export VLLM_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
```

**首次使用**：
- 模型会自动从 HuggingFace 下载到缓存目录（`~/.cache/huggingface/`）

#### 方法 2：使用本地模型路径

**适用场景**：
- 已经手动下载了模型
- 网络访问 HuggingFace 受限
- 需要使用特定版本的模型

**步骤 1**：下载模型

```bash
# 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir /root/models/Qwen2.5-VL-7B-Instruct
```

**步骤 2**：配置路径

在 `app/config.py` 中：

```python
VLLM_MODEL_PATH = "/root/models/Qwen2.5-VL-7B-Instruct"
```

或通过环境变量：

```bash
export VLLM_MODEL_PATH="/root/models/Qwen2.5-VL-7B-Instruct"
```

#### 方法 3：使用 HuggingFace 缓存

如果模型已经通过其他方式下载到 HuggingFace 缓存：

```bash
# 查找缓存位置
ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/

# 配置路径（使用 snapshots 下的具体版本）
VLLM_MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/xxx"
```

### 性能优化配置（RTX 4090D 24GB）

在 `app/config.py` 中：

```python
VLLM_CONFIG = {
    "max_model_len": 8192,              # 最大序列长度
    "gpu_memory_utilization": 0.80,     # GPU 显存利用率 80%
    "tensor_parallel_size": 1,          # 张量并行大小（单 GPU 为 1）
    "dtype": "half",                    # FP16 精度
    "max_num_seqs": 32,                 # 最大并发序列数
}
```

**注意**：`trust_remote_code` 参数在 `llm_utils.py` 中显式传递，**不要**在 `VLLM_CONFIG` 中设置，避免参数重复错误。

### Flash Attention

vLLM **默认启用 Flash Attention**，无需额外配置。Flash Attention 提供：
- ✅ 更快的注意力计算
- ✅ 更低的内存占用
- ✅ 支持更长的序列长度

### 显存优化配置

**如果显存充足（≥24GB）**：
```python
VLLM_CONFIG = {
    "gpu_memory_utilization": 0.85,  # 可以使用更多显存
    "max_model_len": 8192,
}
```

**如果显存紧张（20-24GB）**：
```python
VLLM_CONFIG = {
    "gpu_memory_utilization": 0.75,  # 降低显存使用
    "max_model_len": 4096,           # 降低序列长度
}
```

**如果显存不足（<20GB）**：
```python
VLLM_CONFIG = {
    "dtype": "int8",                 # 使用量化
    "gpu_memory_utilization": 0.70,
    "max_model_len": 2048,
}
```

### 启用量化（可选）

如果需要更低的显存占用：

```python
VLLM_CONFIG = {
    # ... 其他配置
    "quantization": "awq",  # 或 "gptq", "squeezellm"
}
```

---

## 模型加载问题解决

### 🐛 问题 1：模型路径错误

**错误信息**：
```
❌ vLLM 模型加载失败：Qwen/Qwen2.5-8B-Instruct 不是本地文件夹，也不是 'https://huggingface.co/models' 上列出的有效型号标识符
```

**可能原因**：
1. 网络问题：无法访问 `huggingface.co`
2. 模型名称错误：模型 ID 不正确
3. 权限问题：需要认证的私有模型
4. vLLM 版本问题：某些版本的 vLLM 可能不支持某些模型

**解决方案**：

#### 方案 1：使用本地模型（推荐，如果网络有问题）

**步骤 1**：手动下载模型

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型到本地
huggingface-cli download Qwen/Qwen2.5-8B-Instruct \
    --local-dir /root/models/Qwen2.5-8B-Instruct \
    --local-dir-use-symlinks False
```

**步骤 2**：配置本地路径

在 `app/config.py` 中修改：

```python
VLLM_MODEL_PATH = "/root/models/Qwen2.5-8B-Instruct"
```

或者通过环境变量：

```bash
export VLLM_MODEL_PATH="/root/models/Qwen2.5-8B-Instruct"
```

#### 方案 2：使用 HuggingFace 镜像源

如果网络可以访问镜像源，设置环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_MODEL_PATH="Qwen/Qwen2.5-8B-Instruct"
```

**注意**：vLLM 可能不会自动使用 `HF_ENDPOINT` 环境变量，建议使用方案 1（本地模型）。

#### 方案 3：使用其他模型名称

如果 `Qwen/Qwen2.5-8B-Instruct` 不可用，可以尝试：

```python
# 在 app/config.py 中
VLLM_MODEL_PATH = "Qwen/Qwen2-8B-Instruct"  # 备选模型
# 或
VLLM_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # 更小的模型
```

#### 方案 4：使用 HuggingFace 缓存

如果之前已经下载过模型，查找缓存位置：

```bash
# 查找 HuggingFace 缓存
find ~/.cache/huggingface -name "*Qwen*" -type d

# 找到后，使用完整路径
# 例如：~/.cache/huggingface/hub/models--Qwen--Qwen2.5-8B-Instruct/snapshots/xxxxx
```

然后在 `app/config.py` 中设置：

```python
VLLM_MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-8B-Instruct/snapshots/xxxxx"
```

### 详细步骤：手动下载模型

#### 方法 1：使用 huggingface-cli

```bash
# 1. 安装工具
pip install huggingface_hub

# 2. 登录（如果需要）
huggingface-cli login

# 3. 下载模型
huggingface-cli download Qwen/Qwen2.5-8B-Instruct \
    --local-dir /root/models/Qwen2.5-8B-Instruct \
    --local-dir-use-symlinks False

# 4. 配置路径
export VLLM_MODEL_PATH="/root/models/Qwen2.5-8B-Instruct"
```

#### 方法 2：使用 Python 脚本

创建 `download_model.py`：

```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="Qwen/Qwen2.5-8B-Instruct",
    local_dir="/root/models/Qwen2.5-8B-Instruct",
    local_dir_use_symlinks=False
)
print(f"模型已下载到: {model_path}")
```

运行：
```bash
python download_model.py
```

#### 方法 3：使用 git-lfs（如果已安装）

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-8B-Instruct /root/models/Qwen2.5-8B-Instruct
```

### 验证模型路径

下载完成后，验证模型路径：

```bash
# 检查模型文件是否存在
ls -lh /root/models/Qwen2.5-8B-Instruct/

# 应该看到以下文件：
# - config.json
# - tokenizer.json
# - model-*.safetensors 或 pytorch_model.bin
# - 等等
```

---

## 参数配置问题修复

### 🐛 问题 2：参数重复错误

**错误信息**：
```
❌ vLLM 模型初始化失败：langchain_community.llms.vllm.VLLM() 获得了多个关键字参数 'trust_remote_code' 的值
```

**问题原因**：

`trust_remote_code` 参数被传递了两次：
1. 在 `app/llm_utils.py` 中：显式传递 `trust_remote_code=True`
2. 在 `app/config.py` 的 `VLLM_CONFIG` 中：通过 `**config` 展开时也传递了 `trust_remote_code`

当使用 `**config` 展开配置字典时，如果配置中已经包含 `trust_remote_code`，而函数调用时又显式传递了该参数，就会导致参数重复的错误。

**解决方案**：

从 `VLLM_CONFIG` 中移除 `trust_remote_code`，只在 `llm_utils.py` 中显式传递一次。

**修改前**：

`app/config.py`:
```python
VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.85,
    "tensor_parallel_size": 1,
    "dtype": "half",
    "trust_remote_code": True,  # ❌ 这里设置了
    # ...
}
```

`app/llm_utils.py`:
```python
llm = VLLM(
    model=model_path,
    trust_remote_code=True,  # ❌ 这里又设置了
    **config  # ❌ 展开时又传递了一次
)
```

**修改后**：

`app/config.py`:
```python
VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.85,
    "tensor_parallel_size": 1,
    "dtype": "half",
    # "trust_remote_code": True,  # ✅ 已移除，避免重复
    # ...
}
```

`app/llm_utils.py`:
```python
llm = VLLM(
    model=model_path,
    trust_remote_code=True,  # ✅ 只在这里设置一次
    **config  # ✅ 展开时不会重复
)
```

### 最佳实践

当使用 `**kwargs` 展开配置字典时，需要注意：

1. **避免参数重复**：不要在配置字典和函数调用中同时设置同一个参数
2. **显式参数优先**：如果需要在多个地方设置参数，优先在函数调用时显式传递
3. **配置分离**：将需要显式控制的参数从配置字典中分离出来

---

## 性能优化

### 显存优化

**当前配置（RTX 4090D 24GB）**：

```python
VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.80,  # 留出 20% 余量
    "tensor_parallel_size": 1,
    "dtype": "half",  # FP16
}
```

**显存分配**（7B 模型）：
- 模型权重：~14GB (FP16)
- KV Cache：~2-3GB
- 激活值：~2-3GB
- **总计**：~18-20GB

**多模态模型（VL）额外显存**：
- 图像编码器：~2-3GB
- **总计**：~20-24GB

### 推理速度优化

1. **使用更小的模型**（如果不需要多模态）
   ```python
   VLLM_MODEL_PATH = "Qwen/Qwen2.5-4B-Instruct"
   ```

2. **降低序列长度**
   ```python
   "max_model_len": 4096  # 从 8192 降低
   ```

3. **减少图像数量**
   - 在视频帧提取时减少帧数
   - 只提取关键帧

### 首次加载优化

1. **预下载模型**
   ```bash
   ./download_model.sh
   ```

2. **使用本地路径**
   ```python
   VLLM_MODEL_PATH = "/root/models/Qwen2.5-VL-7B-Instruct"
   ```

---

## 故障排除

### 问题 1: vLLM 导入失败

```
ImportError: cannot import name 'VLLM' from 'langchain_community.llms'
```

**解决方案**:
```bash
pip install --upgrade langchain-community vllm
```

### 问题 2: CUDA 不可用

```
RuntimeError: CUDA 不可用
```

**解决方案**:
1. 检查 GPU: `nvidia-smi`
2. 检查 CUDA: `nvcc --version`
3. 重新安装 PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### 问题 3: 显存不足

```
OutOfMemoryError: CUDA out of memory
```

**解决方案**:
1. 降低 `max_model_len`（如 4096）
2. 降低 `gpu_memory_utilization`（如 0.75）
3. 使用量化: `"dtype": "int8"` 或 `"quantization": "awq"`
4. 检查是否有其他进程占用 GPU 显存：`nvidia-smi`
5. 清理 GPU 显存：`kill -9 <占用显存的进程PID>`

### 问题 4: 模型加载慢

首次加载模型需要下载和初始化，可能需要几分钟。这是正常现象。

**优化建议**：
- 预下载模型到本地
- 使用本地路径配置

### 问题 5: 下载失败

```bash
# 检查网络连接
curl -I https://huggingface.co

# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 6: 路径不存在

```bash
# 检查路径
ls -la /root/models/Qwen2.5-8B-Instruct/

# 如果不存在，重新下载
```

### 问题 7: 权限问题

```bash
# 检查权限
ls -ld /root/models/

# 修复权限
chmod -R 755 /root/models/
```

### 问题 8: 多模态输入限制

vLLM 当前版本**对多模态支持有限**。代码中已添加提示，说明图像信息通过文本描述传递。

如果需要图像支持，可以考虑：
- 使用多模态模型（如 Qwen2.5-VL）
- 预处理图像为文本描述
- 使用其他支持图像的推理框架

---

## 回退到 Ollama

如果需要回退到 Ollama：

1. **取消注释 `app/llm_utils.py` 中的 Ollama 代码**
   - 取消注释第 7-17 行（Ollama 导入）
   - 取消注释第 130-155 行（Ollama 函数）

2. **注释掉 vLLM 相关代码**
   - 注释第 19-28 行（vLLM 导入）
   - 注释第 157-305 行（vLLM 函数）

3. **在 `app/config.py` 中**
   - 注释 `VLLM_MODEL_PATH` 和 `VLLM_CONFIG`
   - 取消注释 `MODEL = "qwen3:8b"`

4. **在 `requirements.txt` 中**
   - 确保 `ollama` 已安装

5. **重新安装依赖**:
   ```bash
   pip install ollama
   ```

6. **启动 Ollama 服务**:
   ```bash
   ollama serve
   ```

---

## 📚 参考资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [LangChain vLLM 集成](https://python.langchain.com/docs/integrations/llms/vllm)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [HuggingFace 模型下载指南](https://huggingface.co/docs/hub/models-downloading)
- [Qwen 模型页面](https://huggingface.co/Qwen)

---

## 📝 代码变更摘要

### 主要文件修改

1. **`app/llm_utils.py`**
   - 注释 Ollama 导入和函数
   - 取消注释 vLLM 导入和初始化
   - 重写 `get_response()` 使用 vLLM
   - 更新 `contextualize_query()` 使用 vLLM

2. **`app/config.py`**
   - 取消注释 `VLLM_MODEL_PATH` 配置
   - 取消注释 `VLLM_CONFIG` 配置字典
   - **注意**：不要在其中设置 `trust_remote_code`

3. **`requirements.txt`**
   - 确保 `vllm>=0.6.0` 已安装

---

*最后更新: 2025-01-10*

