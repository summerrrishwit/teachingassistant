# 📚 功能文档

本文档详细说明项目的各项功能实现。

## 目录

1. [RAG功能实现](#rag功能实现)
2. [时间戳问答持久化](#时间戳问答持久化)
3. [完整视频总结](#完整视频总结)

---

## RAG功能实现

### 概述

本项目已成功集成基于 LangChain 的 RAG（检索增强生成）功能，用于提升视频问答的准确性和相关性。

### 主要特性

- **语义检索**: 基于向量相似度检索相关视频片段
- **时间戳保留**: 检索结果包含原始时间戳信息
- **持久化索引**: 支持索引的保存和加载，避免重复构建
- **智能缓存**: 基于视频签名的索引缓存机制

### 技术实现

#### 依赖库

```txt
langchain
langchain-community
langchain-huggingface
faiss-cpu
sentence-transformers
```

#### 核心类：VideoRAGSystem

位置：`app/rag_utils.py`

**主要方法**：

1. **`build_vector_store(segments, video_id)`**
   - 将 Whisper 转录片段转换为 Document 对象
   - 使用 RecursiveCharacterTextSplitter 分割文本
   - 构建 FAISS 向量索引
   - 持久化保存索引

2. **`load_vector_store(video_id)`**
   - 从磁盘加载已保存的索引
   - 自动检查文件完整性
   - 文件不存在时静默返回 False

3. **`retrieve_relevant_context(query, top_k=3)`**
   - 基于查询问题检索最相关的片段
   - 返回包含文本、时间戳、相似度分数的结果

4. **`cleanup_invalid_indices(keep_signatures)`**
   - 清理无效或过期的索引文件
   - 支持保留指定签名的索引

### 工作流程

```
视频转录
    ↓
转换为Document对象（保留时间戳）
    ↓
文本分割（chunk_size=500, overlap=50）
    ↓
生成向量嵌入
    ↓
构建FAISS索引
    ↓
持久化保存（基于视频签名）
    ↓
查询时检索Top-K相关片段
    ↓
合并到LLM提示词中
```

### 使用示例

```python
from app.rag_utils import get_rag_system
from app.workflows import ensure_vector_index

# 确保向量索引可用
segments = whisper_model.transcribe(video_path)
ensure_vector_index(segments, video_path)

# 检索相关上下文
rag_system = get_rag_system()
contexts = rag_system.retrieve_relevant_context(
    query="视频中讲了什么？",
    top_k=3
)

# 使用检索结果
for ctx in contexts:
    print(f"[{ctx['timestamp']}] {ctx['text']}")
    print(f"相似度: {ctx['score']:.2f}")
```

### 配置参数

- **chunk_size**: 500 字符
- **chunk_overlap**: 50 字符
- **top_k**: 默认返回3个最相关片段
- **score_threshold**: L2距离阈值 1.5

### 索引管理

索引文件保存在 `runtime/faiss_index_{video_signature}/` 目录下：
- `index.faiss`: FAISS 向量索引文件
- `index.pkl`: 文档存储和元数据

视频签名基于文件大小和修改时间生成，确保同一视频使用相同索引。

---

## 时间戳问答持久化

### 功能概述

为特定时间戳问答功能添加了数据保留机制，避免用户在不同模式间切换时丢失已生成的问答数据。

### 核心特性

1. **数据保留**
   - QA数据保存到 session state
   - 对话历史完整保留
   - 状态管理（满意度、时间戳等）

2. **智能检测**
   - 自动检测已有数据
   - 避免重复生成
   - 显示已保存数据的时间戳信息

3. **便捷操作**
   - 🔄 重新生成：清除当前数据，允许重新输入
   - 🗑️ 清除问答：清除所有QA相关数据
   - 📥 导出问答：导出为Markdown文件

### Session State 数据结构

```python
st.session_state.qa_timestamp = manual_time          # 时间戳（秒）
st.session_state.qa_transcript = transcript_snippet # 转录片段
st.session_state.qa_full_transcript = full_transcript # 完整转录
st.session_state.qa_summarized_transcript = summarized_transcript # 摘要
st.session_state.qa_conversation_history = []       # 对话历史
```

### 对话历史格式

```python
{
    'timestamp': 90.5,                    # 时间戳（秒）
    'timestamp_display': '1:30',          # 显示格式
    'question': '这个时间点讲了什么？',    # 问题
    'answer': '...',                      # 回答
    'transcript_snippet': '...'           # 相关转录片段
}
```

### 使用流程

#### 首次使用
1. 选择"特定时间戳问答"模式
2. 输入时间戳和问题
3. 系统生成转录和问答界面
4. 进行问答对话

#### 再次访问
1. 选择"特定时间戳问答"模式
2. 系统自动检测并显示已保存的数据
3. 可以直接查看转录片段和对话历史
4. 可以选择重新生成或导出数据

#### 模式切换
- 在完整视频总结和特定时间戳问答间自由切换
- 数据得到完整保留
- 不会丢失任何已生成的内容

### 导出功能

导出的Markdown格式：

```markdown
# 时间戳问答报告

## 问答 1

**时间戳:** 1:30 (90 秒)

**问题:** 这个时间点讲了什么？

**相关转录片段:**
...

**回答:**
...
```

---

## 完整视频总结

### 功能概述

对整个视频进行综合分析，生成结构化的分析报告。

### 分析内容

1. **Video Overview**: 视频概述
2. **Main Topics**: 主要话题和概念
3. **Key Insights**: 重要洞察和学习要点
4. **Structure Analysis**: 内容组织结构分析
5. **Target Audience**: 目标受众分析
6. **Learning Objectives**: 学习目标总结

### 处理流程

```
上传视频
    ↓
完整转录（Whisper）
    ↓
提取关键帧（5帧，均匀分布）
    ↓
构建向量索引（可选）
    ↓
LLM综合分析
    ↓
生成结构化报告
    ↓
展示结果（可导出）
```

### 关键帧提取策略

- 数量：5帧
- 分布：均匀分布在视频时间轴上
- 用途：提供视觉上下文，增强理解

### 结果展示

- 使用 Expander 组织内容
- 支持展开/折叠查看
- 提供导出和重新生成选项

---

## 技术细节

### 多模态融合

系统结合三种信息源：
1. **音频转录**: Whisper 生成的文本
2. **视觉帧**: OpenCV 提取的视频帧
3. **语义检索**: RAG 检索的相关片段

### 提示词工程

不同任务使用不同的提示词模板：
- `video_qa`: 基础问答
- `video_qa_rag`: RAG增强问答
- `video_summary`: 完整视频总结
- `bullet_points`: 要点提取
- `qa_style`: 问答对生成

### 错误处理

- 自动重试机制（Ollama连接）
- 降级方案（stream失败时使用generate）
- 友好的错误提示
- 日志记录

---

## 性能优化

### 缓存策略

- Whisper模型：使用 `@st.cache_resource`
- Embedding模型：使用 `@st.cache_resource`
- 转录结果：基于视频签名缓存
- 向量索引：持久化到磁盘

### 异步处理

- 视频处理可以异步化（未来优化）
- 使用进度条显示处理状态
- 流式输出LLM响应

---

## 常见问题

### Q: RAG索引构建失败怎么办？

A: 系统会自动检测并重建索引。如果持续失败，检查：
- 磁盘空间是否充足
- 转录是否成功
- 查看日志文件获取详细错误信息

### Q: 如何清理旧的索引文件？

A: 使用清理功能：
```python
from app.rag_utils import get_rag_system
rag_system = get_rag_system()
rag_system.cleanup_invalid_indices()
```

### Q: 对话历史会保存多久？

A: 对话历史保存在 session state 中，仅在当前会话有效。刷新页面或关闭浏览器会丢失。

### Q: 支持哪些视频格式？

A: 支持 MP4、WebM、MOV 格式，最大文件大小 200MB。

---

*最后更新: 2025-12-14*

