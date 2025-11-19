# RAG 功能实现说明

## 概述

本项目已成功集成基于 LangChain 的 RAG（检索增强生成）功能，用于提升视频问答的准确性和相关性。

## 主要改进

### 1. 新增依赖

在 `requirements.txt` 中添加了以下依赖：
- `langchain` - LangChain 核心库
- `langchain-community` - LangChain 社区扩展
- `faiss-cpu` - FAISS 向量索引库（CPU版本）
- `sentence-transformers` - 句子嵌入模型

### 2. 新增文件

#### `app/rag_utils.py`
实现了 `VideoRAGSystem` 类，提供以下功能：
- **文档转换**: 将 Whisper 转录片段转换为 LangChain Document 对象，保留时间戳元数据
- **文本分割**: 使用 `RecursiveCharacterTextSplitter` 将长文本分割为合适的 chunk
- **向量索引**: 使用 FAISS 构建向量索引，支持持久化存储
- **语义检索**: 根据问题检索最相关的视频片段（Top-K）
- **时间戳过滤**: 支持结合时间戳和语义检索的混合检索

### 3. 修改的文件

#### `app/llm_utils.py`
- 添加了 `get_rag_system()` 函数，使用 Streamlit 缓存机制
- 修改了 `get_response()` 函数，添加了 `segments` 和 `use_rag` 参数
- 在问答模式下，自动使用 RAG 检索相关上下文
- 将检索到的上下文合并到提示词中，提升回答质量

#### `app/config.py`
- 新增了 `PROMPT_QA_RAG` 提示词模板，专门用于 RAG 增强的问答
- 在 `prompt_dict` 中添加了 `video_qa_rag` 键

#### `app/main.py`
- 在视频转录后自动构建向量索引
- 在问答时传递 `segments` 参数以启用 RAG
- 上传新视频时自动清除旧的向量索引
- 清除问答数据时同步清除向量索引状态

## 工作流程

### 1. 向量索引构建

当用户进行时间戳问答时：
1. 使用 Whisper 转录视频，获得带时间戳的片段
2. 将片段转换为 LangChain Document 对象
3. 使用 `sentence-transformers/all-MiniLM-L6-v2` 模型生成嵌入向量
4. 构建 FAISS 向量索引并保存到 `runtime/faiss_index/`

### 2. 语义检索

当用户提问时：
1. 使用相同 embedding 模型将问题转换为向量
2. 在 FAISS 索引中搜索最相似的 Top-3 片段
3. 返回包含文本、时间戳和相似度分数的结果

### 3. 上下文融合

检索到的上下文会被：
1. 格式化为带时间戳的文本片段
2. 合并到提示词的 `retrieved_contexts` 字段
3. 与局部转录片段和全局摘要一起传递给 LLM

## 时间戳的作用

时间戳在 RAG 系统中扮演着关键角色，提供了**时间维度的上下文信息**，使得检索结果不仅语义相关，还能精确定位到视频中的具体位置。

### 1. 时间戳的存储与保留

在向量索引构建过程中，每个文档片段都会保留完整的时间戳信息：

```python
# app/rag_utils.py - create_documents_from_segments()
doc = Document(
    page_content=seg['text'],
    metadata={
        'start_time': seg['start'],      # 开始时间（秒）
        'end_time': seg['end'],          # 结束时间（秒）
        'timestamp': f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"  # 格式化时间戳
    }
)
```

**作用**：
- **精确定位**：每个检索结果都包含其在视频中的精确时间位置
- **时间范围**：`start_time` 和 `end_time` 定义了片段的完整时间范围
- **可读性**：格式化的 `timestamp`（如 "5:30"）便于用户理解

### 2. 时间戳在检索结果中的呈现

检索到的相关上下文会包含时间戳信息：

```python
# app/rag_utils.py - retrieve_relevant_context()
contexts.append({
    'text': doc.page_content,
    'start_time': doc.metadata.get('start_time', 0),
    'end_time': doc.metadata.get('end_time', 0),
    'timestamp': doc.metadata.get('timestamp', ''),
    'score': float(similarity_score)
})
```

**作用**：
- **可追溯性**：用户可以知道每个相关片段在视频中的位置
- **验证答案**：时间戳帮助用户验证答案的准确性
- **导航支持**：用户可以直接跳转到相关时间点观看视频

### 3. 时间戳在提示词中的融合

检索到的上下文会被格式化为带时间戳的文本，传递给 LLM：

```python
# app/llm_utils.py
retrieved_info = "\n\n".join([
    f"[时间戳: {ctx['timestamp']}] {ctx['text']}"
    for ctx in retrieved_contexts
])
```

**作用**：
- **上下文增强**：LLM 可以看到每个片段的时间位置，生成更准确的回答
- **引用支持**：LLM 可以在回答中引用具体时间戳（如 "在 5:30 处提到..."）
- **时间顺序理解**：帮助 LLM 理解内容的时间顺序和逻辑关系

### 4. 混合检索策略

系统支持结合时间戳和语义检索的混合检索策略：

```python
# app/rag_utils.py - retrieve_around_timestamp()
def retrieve_around_timestamp(self, timestamp: float, window: int = 5, query: str = None):
    # 先进行语义检索
    semantic_results = self.retrieve_relevant_context(query, top_k=5)
    # 然后过滤出时间戳附近的片段
    time_filtered = [
        ctx for ctx in semantic_results
        if abs(ctx['start_time'] - timestamp) <= window * 2
    ]
```

**作用**：
- **精确检索**：当用户指定时间戳时，优先返回该时间点附近的相关内容
- **相关性平衡**：既保证语义相关性，又保证时间相关性
- **上下文聚焦**：在用户关注的时间窗口内进行检索，提高答案的针对性

### 5. 时间戳的实际应用场景

#### 场景 1：用户指定时间戳提问
- **输入**：用户输入 "5:30" 并提问
- **处理**：系统检索语义相关的内容，并优先返回 5:30 附近的内容
- **输出**：回答中包含时间戳引用，如 "在 5:30 处，视频提到..."

#### 场景 2：多轮对话中的时间定位
- **上下文**：用户之前询问了某个时间点的问题
- **检索**：系统可以结合之前的时间戳信息，检索相关时间段的更多内容
- **输出**：提供更完整的时间上下文

#### 场景 3：答案验证与导航
- **检索结果**：返回多个相关片段，每个都带有时间戳
- **用户操作**：用户可以点击时间戳跳转到视频对应位置
- **验证**：用户可以观看视频验证答案的准确性

### 6. 时间戳的技术优势

1. **元数据保留**：时间戳作为文档元数据存储，不影响向量检索性能
2. **低开销**：时间戳信息占用空间极小，几乎不影响索引大小
3. **灵活查询**：支持纯语义检索、时间窗口检索和混合检索
4. **可扩展性**：易于扩展到多视频场景，每个视频的时间戳独立管理

### 7. 时间戳格式说明

- **存储格式**：`start_time` 和 `end_time` 使用浮点数（秒）
- **显示格式**：`timestamp` 使用 "MM:SS" 格式（如 "5:30"）
- **转换逻辑**：
  ```python
  timestamp = f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
  ```

### 8. 未来优化方向

1. **时间索引优化**：构建基于时间戳的二级索引，加速时间窗口查询
2. **时间序列分析**：分析内容的时间分布，识别关键时间段
3. **跨时间关联**：识别不同时间段之间的关联性
4. **时间感知排序**：在检索结果排序中考虑时间距离因素

## 技术特点

### 优势
- **语义理解**: 使用语义相似度而非简单的关键词匹配
- **时间戳保留**: 检索结果包含原始时间戳，便于定位
- **性能优化**: 
  - 使用 Streamlit 缓存避免重复构建索引
  - FAISS 提供毫秒级的检索速度
  - 支持索引持久化，避免重复计算
- **可扩展性**: 易于扩展到多视频场景和更高级的检索策略

### 配置参数

在 `app/rag_utils.py` 中可调整的参数：
- `chunk_size`: 文本分割的块大小（默认 500 字符）
- `chunk_overlap`: 块之间的重叠（默认 50 字符）
- `top_k`: 检索返回的片段数量（默认 3）
- `score_threshold`: 相似度阈值（L2距离，默认 1.5）

## 使用说明

### 首次使用

1. 安装新依赖：
```bash
pip install -r requirements.txt
```

2. 首次运行时，系统会自动下载 `all-MiniLM-L6-v2` embedding 模型（约 80MB）

3. 进行视频问答时，系统会自动构建向量索引（首次需要一些时间）

### 正常使用

1. 上传视频文件
2. 选择"特定时间戳问答"模式
3. 输入时间戳
4. 系统自动转录并构建向量索引
5. 提问时，系统会自动检索相关片段并生成回答

## 性能说明

- **索引构建时间**: 约 1-5 秒（取决于视频长度和转录片段数量）
- **检索时间**: < 100 毫秒
- **内存占用**: 索引大小约等于转录文本大小的 1-2 倍（向量维度 384）

## 注意事项

1. **索引持久化**: 索引保存在 `runtime/faiss_index/` 目录，上传新视频时会自动清除
2. **模型下载**: 首次使用需要下载 embedding 模型，请确保网络连接正常
3. **GPU 加速**: 如需更快的处理速度，可以安装 `faiss-gpu` 替代 `faiss-cpu`

## 未来优化方向

1. **多视频支持**: 支持跨视频检索和课程库管理
2. **高级检索策略**: 
   - 使用压缩检索器减少上下文长度
   - 实现混合检索（语义 + 关键词）
   - 支持重新排序（reranking）
3. **性能优化**: 
   - 使用 GPU 加速
   - 异步索引构建
   - 增量索引更新

