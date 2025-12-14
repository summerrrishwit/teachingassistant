# 🎥 AI Video Assistant

一个基于多模态AI技术的智能视频分析助手，集成了大语言模型、语音识别、计算机视觉和自然语言处理技术，为用户提供全面的视频内容理解、问答和学习笔记生成功能。

## ✨ 核心功能

- **📝 完整视频总结**: 综合分析整个视频内容，包括概述、主要话题、关键洞察等
- **❓ 时间戳问答**: 在特定时间点提问，获得精准的上下文相关答案
- **🖼️ 关键帧提取**: 自动提取重要画面用于视觉理解
- **📊 多模态分析**: 结合视频、音频和文本信息进行全面分析
- **📥 结果导出**: 支持Markdown格式导出分析结果和问答历史
- **🔍 RAG增强检索**: 基于语义相似度的智能检索，提升问答准确性

## 🚀 快速开始

### 环境要求

- Python 3.12+
- Ollama (用于运行大语言模型)
- FFmpeg (用于视频处理)

### 安装步骤

1. **安装 Ollama 并拉取模型**

```bash
# 安装 Ollama (根据系统选择)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: 下载安装包

# 拉取模型
ollama pull gemma3:4b
```

2. **安装系统依赖**

```bash
# 安装 FFmpeg
# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg

# macOS:
brew install ffmpeg
```

3. **克隆项目并安装Python依赖**

```bash
git clone <repository-url>
cd autodl-tmp
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **运行应用**

```bash
# 激活虚拟环境
source venv/bin/activate

# 设置 HuggingFace 镜像源（可选，解决401错误）
export HF_ENDPOINT=https://hf-mirror.com

# 运行 Streamlit 应用
streamlit run app/main.py
```

或者使用启动脚本：

```bash
./run_streamlit.sh
```

## 📂 项目结构

```
app/
├── __init__.py          # 统一导出
├── config.py            # 配置和提示词
├── core.py              # 基础工具（异常、常量、装饰器、日志等）
├── llm_utils.py         # LLM相关功能
├── main.py              # Streamlit主入口
├── rag_utils.py         # RAG系统实现
├── transcript_utils.py  # 语音转录工具
├── ui.py                # UI组件
├── video_utils.py       # 视频处理工具
└── workflows.py         # 工作流处理

runtime/                 # 运行时文件
├── uploaded_video.mp4   # 上传的视频
├── frames/              # 提取的视频帧
├── faiss_index*/        # FAISS向量索引
└── logs/                # 日志文件
```

## 🛠️ 技术栈

- **前端框架**: Streamlit
- **语音识别**: OpenAI Whisper
- **大语言模型**: Ollama (Gemma3:4b)
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

## 📚 文档

- [功能文档](FEATURES.md) - 详细功能说明（RAG、QA持久化等）
- [开发指南](DEVELOPER_GUIDE.md) - 开发相关文档（优化、重构、Bug修复等）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[添加许可证信息]

## 🔗 相关链接

- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
