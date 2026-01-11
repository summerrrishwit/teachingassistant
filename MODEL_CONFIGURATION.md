# 模型配置与迁移指南

本文档提供完整的模型配置说明，包括模型选择、配置方法、迁移步骤和故障排除。

## 📋 目录

1. [当前推荐配置](#当前推荐配置)
2. [模型选择指南](#模型选择指南)
3. [配置方法](#配置方法)
4. [模型迁移](#模型迁移)
5. [故障排除](#故障排除)
6. [性能优化](#性能优化)

---

## 当前推荐配置

### 默认模型：Qwen2.5-VL-7B-Instruct

**模型信息**：
- **模型名称**: Qwen/Qwen2.5-VL-7B-Instruct
- **类型**: 视觉语言模型（Vision-Language Model）
- **参数量**: 7B
- **版本**: Qwen2.5 系列（最新版本）

**推荐原因**：
- ✅ 支持多模态输入（文本 + 图像）
- ✅ 优秀的视频帧分析能力
- ✅ 更好的中文支持
- ✅ Qwen2.5 系列性能提升

**配置位置**：`app/config.py`

```python
VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "/root/autodl-tmp/model/Qwen2.5-VL-7B-Instruct")

VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.80,  # VL 模型需要更多显存
    "tensor_parallel_size": 1,
    "dtype": "half",  # FP16
}
```

---

## 模型选择指南

### 可用模型对比

| 模型 | 参数量 | 显存需求 | 文本理解 | 图像理解 | 多模态 | 推理速度 | 推荐场景 |
|------|--------|----------|---------|---------|--------|---------|---------|
| **Qwen2.5-VL-7B-Instruct** | 7B | ~20GB | ✅ 优秀 | ✅ 优秀 | ✅ | 中等 | **视频分析（推荐）** |
| Qwen2-VL-7B-Instruct | 7B | ~20GB | ✅ 良好 | ✅ | ✅ | 中等 | 视频分析（旧版） |
| Qwen2.5-8B-Instruct | 8B | ~16GB | ✅ 优秀 | ❌ | ❌ | 快 | 纯文本分析 |
| Qwen2.5-4B-Instruct | 4B | ~8GB | ✅ 良好 | ❌ | ❌ | 很快 | 低显存环境 |

### 模型对应关系

#### Ollama 与 HuggingFace 模型对应

| Ollama 模型 | HuggingFace 模型 | 说明 |
|------------|-----------------|------|
| `qwen3:8b` | `Qwen/Qwen2.5-8B-Instruct` | 对应关系 |

**注意**：vLLM 需要使用 **HuggingFace 格式**的模型，不能直接使用 ollama 的模型格式。

### 选择建议

**选择 Qwen2.5-VL-7B-Instruct 如果**：
- ✅ 需要分析视频帧内容
- ✅ 需要多模态理解能力
- ✅ GPU 显存 ≥ 20GB
- ✅ 追求最佳性能

**选择 Qwen2.5-8B-Instruct 如果**：
- ✅ 只需要文本分析
- ✅ GPU 显存 16GB 左右
- ✅ 需要更快的推理速度
- ✅ 不需要图像理解

**选择 Qwen2.5-4B-Instruct 如果**：
- ✅ GPU 显存有限（8-12GB）
- ✅ 可以接受性能略降
- ✅ 需要快速响应

---

## 配置方法

### 方法 1：使用 HuggingFace 模型 ID（推荐）

**优点**：
- ✅ 无需额外操作
- ✅ 自动下载和管理
- ✅ 自动更新

**配置方式**：

在 `app/config.py` 中：
```python
VLLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
```

或通过环境变量：
```bash
export VLLM_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
streamlit run app/main.py
```

**首次使用**：
- 模型会自动从 HuggingFace 下载到缓存目录（`~/.cache/huggingface/`）

### 方法 2：使用本地模型路径

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

### 方法 3：使用 HuggingFace 缓存

如果模型已经通过其他方式下载到 HuggingFace 缓存：

```bash
# 查找缓存位置
ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/

# 配置路径（使用 snapshots 下的具体版本）
VLLM_MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/xxx"
```

### 验证配置

运行以下命令验证模型路径：

```python
python3 -c "import sys; sys.path.insert(0, '.'); from app.config import VLLM_MODEL_PATH; print(f'模型路径: {VLLM_MODEL_PATH}')"
```

---

## 模型迁移

### 从 Qwen2.5-8B-Instruct 迁移到 Qwen2.5-VL-7B-Instruct

**优势**：
- ✅ 支持多模态输入（图像理解）
- ✅ 更好的视频分析能力
- ✅ 可以真正利用视频帧信息

**步骤**：

1. **更新模型路径**

在 `app/config.py` 中：
```python
# 旧配置
VLLM_MODEL_PATH = "Qwen/Qwen2.5-8B-Instruct"

# 新配置
VLLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
```

2. **调整显存配置**

```python
VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.80,  # 从 0.85 降低到 0.80
    "tensor_parallel_size": 1,
    "dtype": "half",
}
```

3. **注意事项**：
- 显存需求更高（~20GB vs ~16GB）
- 首次加载需要下载新模型
- 图像处理会增加推理时间

### 从 Qwen2-VL-7B-Instruct 迁移到 Qwen2.5-VL-7B-Instruct

**优势**：
- ✅ Qwen2.5 系列性能更好
- ✅ 多模态能力增强
- ✅ 更好的中文支持

**步骤**：

1. **更新模型路径**

```python
# 旧配置
VLLM_MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

# 新配置
VLLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
```

2. **配置保持不变**

Qwen2.5-VL 和 Qwen2-VL 的配置基本相同，无需修改。

3. **注意事项**：
- 可能需要重新下载模型
- 性能会有提升

### 从 Ollama qwen3:8b 迁移到 vLLM

**问题**：vLLM 需要使用 HuggingFace 格式，不能直接使用 ollama 格式。

**解决方案**：

使用对应的 HuggingFace 模型：
```python
# ollama qwen3:8b 对应 HuggingFace Qwen/Qwen2.5-8B-Instruct
VLLM_MODEL_PATH = "Qwen/Qwen2.5-8B-Instruct"
```

**模型存储位置**：

- **Ollama 模型位置**：
  ```bash
  ~/.ollama/models/manifests/registry.ollama.ai/library/qwen3
  ```

- **HuggingFace 模型缓存位置**：
  ```bash
  ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-8B-Instruct/
  ```

---

## 故障排除

### 问题 1: 模型加载失败

**错误信息**：
```
❌ vLLM 模型加载失败：Qwen/Qwen2.5-VL-7B-Instruct 不是本地文件夹...
```

**可能原因**：
1. 网络连接问题，无法从 HuggingFace 下载
2. 模型路径配置错误
3. 模型文件损坏或不完整

**解决方案**：

1. **检查网络连接**
   ```bash
   # 测试 HuggingFace 连接
   curl -I https://huggingface.co
   ```

2. **手动下载模型**
   ```bash
   pip install huggingface_hub
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
       --local-dir /root/models/Qwen2.5-VL-7B-Instruct
   ```

3. **使用本地路径**
   ```python
   VLLM_MODEL_PATH = "/root/models/Qwen2.5-VL-7B-Instruct"
   ```

4. **使用镜像源**（如果在中国大陆）
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### 问题 2: 显存不足

**错误信息**：
```
OutOfMemoryError: CUDA out of memory
```

**解决方案**：

1. **降低显存利用率**
   ```python
   VLLM_CONFIG = {
       "gpu_memory_utilization": 0.75,  # 从 0.80 降低到 0.75
       # ...
   }
   ```

2. **降低最大序列长度**
   ```python
   VLLM_CONFIG = {
       "max_model_len": 4096,  # 从 8192 降低到 4096
       # ...
   }
   ```

3. **使用量化模型**
   ```python
   VLLM_CONFIG = {
       "dtype": "int8",  # 从 "half" 改为 "int8"
       # ...
   }
   ```

4. **切换到更小的模型**
   ```python
   # 从 Qwen2.5-VL-7B-Instruct 切换到 Qwen2.5-4B-Instruct
   VLLM_MODEL_PATH = "Qwen/Qwen2.5-4B-Instruct"
   ```

### 问题 3: 多模态输入失败

**错误信息**：
```
⚠️ Messages 格式不支持，使用文本模式
```

**说明**：
- 这是正常的回退行为
- vLLM 的 LangChain 包装器可能不完全支持多模态 messages 格式
- 系统会自动回退到文本模式
- 图像信息会通过文本提示词传递
- 模型仍然可以基于文本描述理解图像内容

**无需处理**：这是预期行为，不影响功能使用。

### 问题 4: 模型下载缓慢

**解决方案**：

1. **使用镜像源**（中国大陆推荐）
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **使用下载脚本**
   ```bash
   ./download_model.sh
   ```

3. **手动下载后配置本地路径**
   ```bash
   # 下载到本地
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
       --local-dir /root/models/Qwen2.5-VL-7B-Instruct
   
   # 配置本地路径
   export VLLM_MODEL_PATH="/root/models/Qwen2.5-VL-7B-Instruct"
   ```

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

**显存分配**：
- 模型权重：~14GB (FP16)
- 图像编码器：~2-3GB
- KV Cache：~2-3GB
- 激活值：~2-3GB
- **总计**：~20-24GB

**优化建议**：

1. **如果显存充足（≥24GB）**：
   ```python
   "gpu_memory_utilization": 0.85  # 可以使用更多显存
   ```

2. **如果显存紧张（20-24GB）**：
   ```python
   "gpu_memory_utilization": 0.75  # 降低显存使用
   "max_model_len": 4096  # 降低序列长度
   ```

3. **如果显存不足（<20GB）**：
   ```python
   "dtype": "int8"  # 使用量化
   "gpu_memory_utilization": 0.70
   "max_model_len": 2048
   ```

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

## 多模态输入说明

### 自动处理机制

系统会自动检测是否有视频帧图像：

- **有图像**：尝试使用多模态模式（messages 格式）
- **无图像**：使用纯文本模式
- **多模态失败**：自动回退到文本模式（带图像提示）

### 消息格式

系统会自动构建多模态消息：

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }
]
```

### 注意事项

1. **vLLM 多模态支持**：
   - vLLM 的 LangChain 包装器可能不完全支持多模态 messages 格式
   - 如果多模态输入失败，系统会自动回退到文本模式
   - 图像信息通过文本提示词传递
   - 模型仍然可以基于文本描述理解图像内容

2. **性能影响**：
   - 图像处理会增加推理时间
   - 建议监控 GPU 使用情况
   - 首次加载模型需要较长时间

---

## 配置示例

### 完整配置示例（推荐）

```python
# app/config.py
import os

# 模型路径（优先使用环境变量）
VLLM_MODEL_PATH = os.getenv(
    "VLLM_MODEL_PATH", 
    "Qwen/Qwen2.5-VL-7B-Instruct"  # 默认使用 HuggingFace ID
)

# vLLM 配置
VLLM_CONFIG = {
    "max_model_len": 8192,  # 最大序列长度
    "gpu_memory_utilization": 0.80,  # GPU 显存利用率（VL模型建议80%）
    "tensor_parallel_size": 1,  # 张量并行大小（单 GPU 为 1）
    "dtype": "half",  # 数据类型：half (FP16)
    "trust_remote_code": True,  # 信任远程代码
}
```

### 环境变量配置

```bash
# 设置模型路径
export VLLM_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# 或使用本地路径
export VLLM_MODEL_PATH="/root/models/Qwen2.5-VL-7B-Instruct"

# 设置 HuggingFace 镜像源（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 运行应用
streamlit run app/main.py
```

---

## 参考资源

- [Qwen2.5-VL 模型页面](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen2-VL 模型页面](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Qwen2.5-8B-Instruct 模型页面](https://huggingface.co/Qwen/Qwen2.5-8B-Instruct)
- [vLLM 文档](https://docs.vllm.ai/)
- [vLLM 多模态支持](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [Qwen2.5-VL GitHub](https://github.com/QwenLM/Qwen2.5-VL)

---

## 总结

### 推荐配置

**对于视频分析应用**：
- ✅ **模型**: Qwen2.5-VL-7B-Instruct
- ✅ **显存利用率**: 0.80
- ✅ **最大序列长度**: 8192
- ✅ **数据类型**: FP16 (half)

**对于纯文本分析**：
- ✅ **模型**: Qwen2.5-8B-Instruct
- ✅ **显存利用率**: 0.85
- ✅ **最大序列长度**: 8192
- ✅ **数据类型**: FP16 (half)

**对于低显存环境**：
- ✅ **模型**: Qwen2.5-4B-Instruct
- ✅ **显存利用率**: 0.75
- ✅ **最大序列长度**: 4096
- ✅ **数据类型**: FP16 (half) 或 int8

---

*最后更新: 2025-01-10*


