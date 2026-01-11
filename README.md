# 🎥 AI Video Assistant

一个基于多模态AI技术的智能视频分析助手，集成了大语言模型、语音识别、计算机视觉和自然语言处理技术，为用户提供全面的视频内容理解、问答和学习笔记生成功能。

## ✨ 核心功能

- **📝 完整视频总结**: 综合分析整个视频内容，包括概述、主要话题、关键洞察等
- **❓ 时间戳问答**: 在特定时间点提问，获得精准的上下文相关答案
- **📌 要点提取**: 提取关键概念和要点，生成结构化的学习笔记
- **❓ 问答对生成**: 将视频内容转换为问答对格式，便于学习和复习
- **🖼️ 关键帧提取**: 自动提取重要画面用于视觉理解
- **📊 多模态分析**: 结合视频、音频和文本信息进行全面分析
- **🔍 RAG增强检索**: 基于语义相似度的智能检索，提升问答准确性
- **📥 结果导出**: 支持Markdown格式导出分析结果和问答历史

## 🚀 快速开始

### 环境要求

- **Python**: 3.12+
- **GPU**: NVIDIA GPU（推荐 RTX 4090D 24GB 或更高）
- **CUDA**: CUDA 11.8+ 或 12.1+
- **显存**: 建议 24GB+（对于 Qwen2.5-VL-7B-Instruct 模型）
- **FFmpeg**: 用于视频处理

### 安装步骤

2. **创建虚拟环境并安装依赖**

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **安装系统依赖（FFmpeg）**

```bash
# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg
# macOS:
brew install ffmpeg
```

3. **配置模型（可选）**

模型会在首次使用时自动下载。如果需要手动下载：

```bash
# 使用下载脚本
python download_model.py

# 或使用环境变量设置模型路径
export VLLM_MODEL_PATH="/path/to/local/model"
```

4. **运行应用**

**方式 1：使用启动脚本（推荐）**

```bash
./run_streamlit.sh
```

**方式 2：手动运行**

```bash
# 设置 HuggingFace 镜像源（可选，解决网络问题）
export HF_ENDPOINT=https://hf-mirror.com

# 设置 PYTHONPATH（确保可以导入 app 模块）
export PYTHONPATH=/root/autodl-tmp:$PYTHONPATH

# 运行 Streamlit 应用
streamlit run app/main.py
```

**方式 3：直接运行**

```bash
python -m streamlit run app/main.py
```

## 📂 项目结构

```
app/
├── __init__.py              # 统一导出常用内容
├── config.py                # 配置（vLLM配置、路径配置等）
├── core.py                  # 基础工具（异常、常量、装饰器、日志、验证器等）
├── llm_utils.py             # LLM相关功能（vLLM模型管理、响应生成）
├── main.py                  # Streamlit主入口
├── prompts.py               # 提示词模板定义
├── rag_utils.py             # RAG系统实现（向量索引、检索）
├── transcript_utils.py     # 语音转录工具（Whisper）
├── ui.py                    # UI组件（侧边栏、头部、卡片等）
├── video_utils.py           # 视频处理工具（帧提取）
├── workflows.py             # 工作流处理（总结、问答、要点提取等）
├── services/                # 服务层
│   ├── __init__.py
│   └── vector_service.py    # 向量索引服务
├── models/                  # 数据模型层
│   ├── __init__.py
│   └── workflow_models.py   # 工作流数据模型
├── utils/                   # 工具函数层
│   ├── __init__.py
│   └── timestamp.py        # 时间戳工具函数
└── styles/                  # 样式模块
    ├── __init__.py
    └── css.py               # CSS样式定义

runtime/                     # 运行时文件
├── uploaded_video.mp4      # 上传的视频
├── frames/                  # 提取的视频帧
├── faiss_index*/            # FAISS向量索引（按视频签名存储）
└── logs/                    # 日志文件
```

### 代码结构说明

项目采用模块化设计，主要分为以下几个层次：

- **核心层** (`core.py`): 提供基础工具类、异常、常量、装饰器等
- **服务层** (`services/`): 业务逻辑服务，如向量索引服务
- **工具层** (`utils/`): 通用工具函数，如时间戳解析
- **模型层** (`models/`): 数据模型定义
- **样式层** (`styles/`): UI样式定义

### 导入方式

项目统一使用绝对导入：

```python
from app.config import VIDEO_PATH, FRAME_DIR
from app.services.vector_service import clear_vector_index_state
from app.utils.timestamp import parse_timestamp
from app.styles.css import MAIN_CSS
```

## 🛠️ 技术栈

- **前端框架**: Streamlit
- **语音识别**: OpenAI Whisper
- **大语言模型**: 支持 **Ollama** 和 **vLLM** 双模式部署
  - **Ollama 模式**（默认）: 本地部署，支持多模态，易于使用
  - **vLLM 模式**: 高性能推理，支持 Flash Attention，需要 GPU
- **向量数据库**: FAISS
- **视频处理**: OpenCV
- **RAG框架**: LangChain
- **嵌入模型**: sentence-transformers/all-MiniLM-L6-v2

## 💡 使用说明

### 完整视频总结模式

1. 上传视频文件（支持 MP4、WebM、MOV，最大200MB）
2. 点击"🚀 开始完整分析"
3. 系统将：
   - 使用 Whisper 转录整个视频
   - 提取5个关键帧
   - 构建向量索引（可选）
   - 使用 LLM 生成综合分析
4. 查看分析结果，可导出为 Markdown

### 时间戳问答模式

1. 上传视频文件
2. 点击"🎯 开始时间戳问答"
3. 输入时间戳（格式：HH:MM:SS、MM:SS 或 SS）
4. 输入问题
5. 系统将：
   - 提取时间戳附近的转录片段
   - 提取时间戳附近的视频帧
   - 使用 RAG 检索相关上下文
   - 使用 LLM 生成答案
6. 支持多轮对话，可导出问答历史

### 要点提取模式

1. 上传视频文件
2. 点击"📋 开始提取要点"
3. 系统将自动提取关键概念和要点
4. 生成结构化的学习笔记

### 问答对生成模式

1. 上传视频文件
2. 点击"📝 开始生成问答对"
3. 系统将视频内容转换为问答对格式
4. 适合学习和复习使用

## ⚙️ 配置说明

### 双模式部署

项目支持两种 LLM 部署模式，可根据需求选择：

#### 模式 1: Ollama 部署（默认，推荐）

**特点**：
- ✅ 易于安装和使用
- ✅ 支持多模态输入（图像 + 文本）
- ✅ 本地部署，无需 GPU（CPU 也可运行）
- ✅ 模型管理简单（`ollama pull` 即可）

**配置方法**：

1. **安装 Ollama 服务**：
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **下载模型**：
   ```bash
   # 推荐模型（支持多模态）
   ollama pull qwen2.5-vl:7b
   # 或使用纯文本模型（更快）
   ollama pull qwen3:8b
   ```

3. **配置模型名称**（`app/config.py`）：
   ```python
   MODEL = "qwen2.5-vl:7b"  # 或 "qwen3:8b"
   ```

4. **启动 Ollama 服务**：
   ```bash
   ollama serve
   ```

**模型对应关系**：
- `qwen3:8b` → `Qwen/Qwen2.5-8B-Instruct`（纯文本）
- `qwen2.5-vl:7b` → `Qwen/Qwen2.5-VL-7B-Instruct`（多模态）

#### 模式 2: vLLM 部署（高性能）

**特点**：
- ✅ 高性能推理（Flash Attention 加速）
- ✅ 支持连续批处理，并发能力强
- ✅ 显存效率高（PagedAttention）
- ⚠️ 需要 NVIDIA GPU 和 CUDA
- ⚠️ 仅支持 Linux（CUDA 环境）

**配置方法**：

1. **安装依赖**：
   ```bash
   pip install vllm>=0.6.0
   ```

2. **配置模型路径**（`app/config.py`）：
   ```python
   # 取消注释 vLLM 配置
   VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
   
   VLLM_CONFIG = {
       "max_model_len": 8192,
       "gpu_memory_utilization": 0.80,
       "tensor_parallel_size": 1,
       "dtype": "half",  # FP16
   }
   ```

3. **切换代码**（`app/llm_utils.py`）：
   - 注释 Ollama 相关代码
   - 取消注释 vLLM 相关代码

**环境要求**：
- NVIDIA GPU（推荐 RTX 4090D 24GB 或更高）
- CUDA 11.8+ 或 12.1+
- 显存：建议 24GB+（对于 7B 模型）

### 模式对比

| 特性 | Ollama | vLLM |
|------|--------|------|
| **安装难度** | ⭐ 简单 | ⭐⭐⭐ 需要 CUDA |
| **推理速度** | 中等 | **快** |
| **显存效率** | 中等 | **高**（PagedAttention） |
| **并发处理** | 有限 | **优秀**（连续批处理） |
| **Flash Attention** | ❌ | ✅ **默认启用** |
| **多模态支持** | ✅ **完整支持** | ⚠️ 部分支持 |
| **平台支持** | macOS/Linux/Windows | Linux（CUDA） |
| **GPU 要求** | 可选（CPU 也可） | **必需** |
| **模型格式** | Ollama 格式 | HuggingFace 格式 |

### 模式切换

**从 Ollama 切换到 vLLM**：

1. 在 `app/llm_utils.py` 中：
   - 注释 Ollama 导入和函数（第 7-17 行，第 130-155 行）
   - 取消注释 vLLM 导入和函数（第 19-28 行，第 157-305 行）

2. 在 `app/config.py` 中：
   - 注释 `MODEL = "qwen3:8b"`
   - 取消注释 `VLLM_MODEL_PATH` 和 `VLLM_CONFIG`

3. 更新 `get_response()` 函数调用 vLLM 而非 Ollama

**从 vLLM 切换回 Ollama**：

1. 反向操作上述步骤
2. 确保 Ollama 服务正在运行：`ollama serve`

详细配置说明请参考 [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) 和 [VLLM_MIGRATION.md](VLLM_MIGRATION.md)

### 环境变量

**Ollama 模式**：
```bash
# 无需额外环境变量，只需确保 Ollama 服务运行
ollama serve
```

**vLLM 模式**：
```bash
# 设置模型路径
export VLLM_MODEL_PATH="/path/to/local/model"
# 或使用 HuggingFace 模型 ID
export VLLM_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# 设置 HuggingFace 镜像源（中国大陆推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 设置 HuggingFace Token（如果需要）
export HF_TOKEN=your_token_here
```

## 📚 文档

- [功能文档](FEATURES.md) - 详细功能说明（RAG、QA持久化等）
- [模型配置指南](MODEL_CONFIGURATION.md) - 模型配置、迁移和故障排除
- [RAG优化文档](RAG_UTILS_OPTIMIZATION.md) - RAG系统优化说明

## 🔧 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'app'**
   - 确保从项目根目录运行：`streamlit run app/main.py`
   - 或设置 PYTHONPATH：`export PYTHONPATH=/root/autodl-tmp:$PYTHONPATH`

2. **Ollama 连接失败**
   - 确保 Ollama 服务正在运行：`ollama serve`
   - 检查模型是否已下载：`ollama list`
   - 下载缺失的模型：`ollama pull qwen3:8b`

3. **vLLM 模型加载失败**
   - 检查 GPU 是否可用：`nvidia-smi`
   - 检查 CUDA 版本：`nvcc --version`
   - 检查模型路径是否正确
   - 参考 [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) 的故障排除部分

4. **显存不足（vLLM 模式）**
   - 降低 `gpu_memory_utilization`（如 0.75）
   - 降低 `max_model_len`（如 4096）
   - 使用量化模型（dtype='int8'）
   - 或切换到 Ollama 模式（CPU 也可运行）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献者

感谢以下贡献者：

- [@summerrrishwit](https://github.com/summerrrishwit) - Rongzhi Xia


## 🔗 相关链接

- [vLLM 文档](https://docs.vllm.ai/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [LangChain 文档](https://python.langchain.com/)
- [Qwen2.5-VL 模型](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

---

*最后更新: 2025-01-10*

