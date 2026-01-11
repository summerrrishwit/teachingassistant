# 📚 功能文档

本文档详细说明项目的各项功能实现。

## 目录

1. [RAG功能实现](#rag功能实现)
2. [时间戳问答持久化](#时间戳问答持久化)
3. [完整视频总结](#完整视频总结)
4. [多模态视觉处理](#多模态视觉处理)
5. [RAG与多模态结合](#rag与多模态结合)

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

## 多模态视觉处理

### 概述

本项目充分利用多模态技术，将视频的视觉信息（视频帧）与文本信息（转录文本）相结合，通过支持视觉的多模态LLM（Ollama gemma3:4b）进行综合分析和问答。

### 视觉处理流程

#### 阶段一：视频帧提取

**1. 特定时间戳帧提取** (`extract_frames_around`)

用于时间戳问答场景，提取指定时间点附近的视频帧：

```python
# 提取时间戳 ±2秒 范围内的帧，采样率1fps
frames = extract_frames_around(
    video_path=video_path,
    timestamp=90.0,  # 90秒处
    frame_dir=frame_dir,
    window=2,        # ±2秒窗口
    fps=1            # 每秒1帧
)
```

**处理步骤**：
1. 使用 OpenCV 打开视频文件
2. 计算时间窗口：`[timestamp - window, timestamp + window]`
3. 按指定fps采样提取帧
4. BGR → RGB 颜色空间转换（OpenCV默认BGR，需转换为RGB）
5. 转换为PIL Image对象
6. 保存为JPEG格式（`frame_0.jpg`, `frame_1.jpg`等）

**2. 关键帧提取** (`extract_key_frames_for_summary`)

用于完整视频总结，均匀提取关键帧：

```python
# 从整个视频中均匀提取5个关键帧
summary_frames = extract_key_frames_for_summary(
    video_path=video_path,
    frame_dir=frame_dir,
    num_frames=5
)
```

**提取策略**：
- 数量：默认5帧
- 分布：均匀分布在视频时间轴上
- 文件命名：`summary_frame_0.jpg` 到 `summary_frame_4.jpg`

#### 阶段二：图像编码

将提取的JPEG图像转换为Base64编码，以便通过Ollama API传递：

```python
def image_to_base64(image_path):
    """将图像文件转换为Base64编码字符串"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
```

**编码流程**：
1. 以二进制模式读取JPEG文件
2. Base64编码
3. 解码为UTF-8字符串（Ollama API要求）

#### 阶段三：传递给Ollama

**1. 构建多模态消息**

```python
# 收集所有图像并编码
prompt_inputs = []
for image_path in frame_paths:
    if os.path.exists(image_path):
        prompt_inputs.append(image_to_base64(image_path))

# 构建包含图像的消息
messages.append({
    "role": "user",
    "content": prompt,  # 文本提示词
    "images": prompt_inputs  # Base64编码的图像列表
})
```

**2. 调用Ollama Chat API**

```python
response = ollama.chat(
    model="gemma3:4b",  # 支持视觉的多模态模型
    messages=messages,  # 包含图像的消息列表
    stream=True         # 流式输出
)
```

**3. 降级方案**

如果`ollama.chat`失败，使用`ollama.generate`作为降级：

```python
result = ollama.generate(
    model="gemma3:4b",
    prompt=prompt,
    images=prompt_inputs  # 直接传递图像列表
)
```

### 完整数据流图

```
视频文件 (MP4)
    ↓
[OpenCV] 提取帧 (BGR格式)
    ↓
[PIL] 转换为RGB → JPEG文件
    ↓
[Base64编码] JPEG → Base64字符串
    ↓
[Ollama API] 
    messages = [{
        "role": "user",
        "content": prompt,
        "images": [base64_str1, base64_str2, ...]
    }]
    ↓
[Ollama服务] 
    - 解码base64图像
    - Vision Encoder处理
    - 与文本融合
    - 多模态LLM生成
    ↓
流式返回文本回答
```

### 技术要点

1. **颜色空间转换**：OpenCV使用BGR，需要转换为RGB供PIL和显示使用
2. **图像格式**：统一保存为JPEG格式，平衡质量和文件大小
3. **Base64编码**：Ollama API要求图像以Base64字符串形式传递
4. **批量处理**：支持同时传递多帧图像，提供更丰富的视觉上下文
5. **错误处理**：包含重试机制和降级方案，确保系统稳定性

### 技术细节与最佳实践

#### 1. MP4 → OpenCV 提取帧（BGR）

**处理流程**：

```python
cap = cv2.VideoCapture(mp4_path)
success, frame = cap.read()  # 返回 numpy.ndarray (H×W×3)，BGR格式
```

**关键点**：
- `cv2.VideoCapture()` 解码视频流
- `cap.read()` 每次返回一帧，形状通常是 `H×W×3`
- **默认颜色通道顺序是 BGR**（不是RGB）

**常见问题与解决方案**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 颜色偏蓝/偏红 | OpenCV是BGR，PIL按RGB解释 | 必须进行BGR→RGB转换 |
| 画面横竖不对 | 手机视频带旋转metadata，OpenCV未自动旋转 | 需要额外处理旋转（读取EXIF） |
| 帧数过多 | 按FPS全抽会非常多帧 | 使用抽帧策略：每N秒取1帧，或按场景切换取关键帧 |

**抽帧策略建议**：
- ✅ **优先使用**：每1-2秒取1帧，或按场景切换（scene change）取关键帧
- ✅ **保留时间戳**：方便在prompt里说明顺序
- ❌ **避免**：按原始FPS全抽（会导致base64体积爆炸、请求变慢）

#### 2. OpenCV(BGR) → PIL(RGB) → JPEG

**处理流程**：

```python
# 正确转换方式
rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
img = PIL.Image.fromarray(rgb)
img.save("frame.jpg", "JPEG", quality=75)
```

**关键点**：
- 必须进行 `BGR → RGB` 转换
- 使用PIL保存JPEG（压缩后体积更小，适合base64）

**⚠️ 最常见坑（非常重要）**：

**不要"RGB转完又用OpenCV去写JPEG"**：
- OpenCV的 `cv2.imencode('.jpg', frame)` 默认把输入当BGR处理
- 如果传进去的是RGB，会导致颜色再错一次（变回错误）

**两条路线二选一**：

| 路线 | 处理方式 | 适用场景 |
|------|----------|----------|
| **全程OpenCV** | BGR帧 → `cv2.imencode()`（不做RGB转换） | 不需要PIL的场景 |
| **OpenCV + PIL** | BGR → RGB → PIL保存JPEG | 需要PIL处理的场景（当前项目使用） |

**优化建议**：
- JPEG质量（quality）：**70-85** 就够（不要太高）
- 分辨率缩放：长边缩到 **768/1024** 附近（大幅降低base64体积）
- 文字/屏幕类画面：可适当提高质量或改用PNG（但PNG可能更大）

#### 3. JPEG bytes → Base64 字符串

**Base64编码的作用**：

Base64编码是**必需的**，原因如下：

1. **JSON兼容性**：
   - JPEG是二进制bytes，不能直接放进JSON
   - JSON只能包含文本，需要将二进制安全表示为文本
   - Base64将二进制图片(bytes) → 可放进JSON的纯文本字符串

2. **跨平台传输稳定性**：
   - 不会被编码、转义、换行、字符集搞坏
   - 跨平台/跨语言稳定传输

3. **符合Ollama REST API要求**：
   - Ollama的REST接口在`images`字段里期望的是 `base64-encoded image data`
   - 这是Ollama原生API的标准格式

**处理流程**：

```python
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
```

**⚠️ 常见问题**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Base64体积膨胀 | Base64会膨胀体积（约+33%） | 严格限制帧数（4-12张），先缩放再编码 |
| HTTP body过大 | 多帧叠加后体积很大 | 限制帧数、降低分辨率、控制JPEG质量 |
| 传输慢/推理慢 | 请求体过大 | 先缩放再编码，不要先编码再缩放 |
| 带换行符 | shell base64生成时带换行 | 去掉换行（`tr -d '\n'`） |

**格式要求**：
- ✅ **纯base64字符串**，不要加 `data:image/jpeg;base64,` 前缀
- ✅ Ollama REST API期望的是纯base64数据
- ❌ 不要添加MIME类型前缀

#### 4. Ollama API：messages + images 结构

**标准消息结构**：

```python
messages = [{
    "role": "user",
    "content": prompt,  # 文本提示词
    "images": [base64_str1, base64_str2, ...]  # Base64编码的图像数组
}]
```

**API调用**：

```python
response = ollama.chat(
    model="gemma3:4b",
    messages=messages,
    stream=True
)
```

**关键说明**：
- 这是Ollama原生 `/api/chat` 的标准形态
- 图像base64放在message对象的`images`数组里
- 官方API/vision文档明确支持这种"images array of base64"

**⚠️ 重要注意事项**：

1. **只有视觉模型才支持**：
   - 只有"视觉模型"才会真正看图（如llava、llama vision系列等）
   - **非视觉模型会忽略或报错**（取决于实现）
   - 需要确认使用的模型是否支持视觉输入

2. **OpenAI兼容接口的区别**：
   - 如果走OpenAI兼容接口（`/v1/chat/completions`），图像可能要用`content parts`或`image_url/base64`的另一种结构
   - 当前项目使用的是Ollama原生`messages+images`方式

#### 5. gemma3:4b 视觉支持说明

**当前状态**：

项目配置使用 `gemma3:4b` 模型，但需要注意：

1. **标准Gemma模型不支持视觉**：
   - 标准的Gemma 3模型（包括gemma3:4b）**通常不支持视觉输入**
   - Gemma是纯文本模型，没有视觉编码器（Vision Encoder）

2. **如果使用非视觉模型**：
   - 传递图像会被忽略或导致错误
   - 模型只会基于文本prompt生成回答
   - 视觉信息实际上没有被利用

3. **建议的视觉模型**：
   - **llava系列**：`llava:7b`, `llava:13b`, `llava:34b`
   - **llama-vision系列**：支持视觉的Llama变体
   - **其他视觉模型**：如`bakllava`, `minicpm-v`等

**如何验证模型是否支持视觉**：

```bash
# 检查模型信息
ollama show gemma3:4b

# 尝试传递图像测试
ollama run gemma3:4b "描述这张图片" --images image.jpg
```

**如果模型不支持视觉**：
- 系统仍可工作，但视觉信息会被忽略
- 建议切换到支持视觉的模型（如llava系列）
- 或者移除图像传递，仅使用文本RAG功能

#### 6. 实战优化清单

**帧抽取优化**：
- ✅ 每1-2秒抽1帧，或场景变化抽关键帧（优先少而精）
- ❌ 避免帧太多：会导致"信息冗余 + 费用/延迟上升 + 模型注意力被稀释"

**分辨率与压缩**：
- ✅ 长边缩到768/1024（保留细节又不爆体积）
- ✅ JPEG quality 70-85
- ✅ 文字/屏幕类画面：适当提高质量或改用PNG（但PNG可能更大）

**颜色与编码一致性**：
- ✅ 用PIL保存JPEG：一定先BGR→RGB
- ✅ 用OpenCV imencode：不要做BGR→RGB，直接编码
- ❌ 不要混用两种方式

**Prompt组织**：
- ✅ 多帧时在prompt里明确顺序与含义：
  ```
  "共有N张图，按时间顺序从早到晚"
  "请先概述每张图，再给出整体事件/动作总结"
  ```
- ✅ 让模型知道这是"同一段视频的不同时间点"
- ❌ 否则模型可能当作无关图片拼接

**性能监控指标**：
- 单张图像分辨率/大小
- 一次发送的帧数
- Base64编码后的总大小
- API响应时间

**推荐配置**：
- 帧数：4-12张（根据任务复杂度调整）
- 分辨率：长边768-1024像素
- JPEG质量：75-85
- Base64总大小：控制在2-5MB以内

---

## RAG与多模态结合

### 概述

RAG（检索增强生成）功能与多模态视觉处理深度结合，在提示词层面明确要求模型同时使用图像和RAG检索的文本信息，实现更精准的视频问答。

### RAG增强的提示词设计

#### PROMPT_QA_RAG

专门为RAG增强问答设计的提示词模板：

```python
PROMPT_QA_RAG = """You are a helpful teaching assistant. You will be provided with:
- A few video frames (images)
- A focused transcript snippet around a specific timestamp
- Retrieved relevant contexts from the video (with timestamps, based on semantic similarity)
- A summary of the video as background

Your job is to answer the question using all provided information. When referencing specific content, mention the timestamp if available.

Retrieved Relevant Contexts (语义检索到的相关片段):
{retrieved_contexts}

Global Video Summary:
{global_context}

Focused Transcript Snippet (around timestamp):
{text}

Question: {question}
Answer:"""
```

**关键特点**：
- ✅ 明确告知模型会收到**视频帧图像**
- ✅ 明确告知模型会收到**RAG检索的文本片段**（带时间戳）
- ✅ 要求使用**所有提供的信息**（图像 + 文本）
- ✅ 强调在引用内容时**提及时间戳**

### 执行流程

#### 步骤1：RAG文本检索

```python
# 使用RAG系统检索相关文本片段
retrieved_contexts = rag_system.retrieve_relevant_context(
    query=question,
    top_k=3  # 返回最相关的3个片段
)

# 格式化检索结果
retrieved_info = "\n\n".join([
    f"[时间戳: {ctx['timestamp']}] {ctx['text']}"
    for ctx in retrieved_contexts
])
```

**检索结果示例**：
```
[时间戳: 1:30] 这是RAG检索到的第一段相关文本...
[时间戳: 2:15] 这是RAG检索到的第二段相关文本...
[时间戳: 3:45] 这是RAG检索到的第三段相关文本...
```

#### 步骤2：图像提取与编码

```python
# 提取时间戳附近的视频帧
frames = extract_frames_around(video_path, timestamp, frame_dir)

# 转换为Base64编码
prompt_inputs = []
for image_path in frames:
    prompt_inputs.append(image_to_base64(image_path))
```

#### 步骤3：Prompt构建（融合RAG文本）

```python
# 使用RAG增强的提示词模板
prompt = PROMPT_QA_RAG.format(
    text=transcript_snippet,           # 时间戳附近的转录片段
    question=question,                  # 用户问题
    global_context=summarized_transcript,  # 视频摘要
    retrieved_contexts=retrieved_info  # RAG检索的文本上下文
)
```

#### 步骤4：多模态消息构建

```python
# 构建包含图像和文本的完整消息
messages.append({
    "role": "user",
    "content": prompt,        # 包含RAG检索文本的提示词
    "images": prompt_inputs   # 视频帧的Base64编码
})
```

### 最终消息结构

发送给Ollama的完整消息包含：

```python
{
    "role": "user",
    "content": """You are a helpful teaching assistant. You will be provided with:
- A few video frames (images)
- Retrieved relevant contexts from the video (with timestamps, based on semantic similarity)
...

Retrieved Relevant Contexts (语义检索到的相关片段):
[时间戳: 1:30] RAG检索到的第一段文本...
[时间戳: 2:15] RAG检索到的第二段文本...

Global Video Summary:
视频的完整摘要...

Focused Transcript Snippet (around timestamp):
时间戳附近的转录片段...

Question: 用户的问题
Answer:""",
    "images": [
        "iVBORw0KGgoAAAANSUhEUgAA...",  # 视频帧1的base64
        "iVBORw0KGgoAAAANSUhEUgAA...",  # 视频帧2的base64
        ...
    ]
}
```

### 信息源整合

在RAG增强模式下，模型同时接收四种信息源：

| 信息源 | 类型 | 来源 | 用途 |
|--------|------|------|------|
| 视频帧图像 | 视觉 | OpenCV提取 | 提供视觉上下文 |
| RAG检索文本 | 文本 | 语义向量检索 | 提供相关片段上下文 |
| 时间戳转录 | 文本 | Whisper转录 | 提供局部文本上下文 |
| 视频摘要 | 文本 | 完整转录摘要 | 提供全局背景 |

### 与普通模式的对比

| 模式 | 图像 | RAG文本 | Prompt类型 | 适用场景 |
|------|------|---------|-----------|----------|
| `video_qa` | ✅ | ❌ | `PROMPT_QA` | 基础问答，无需语义检索 |
| `video_qa_rag` | ✅ | ✅ | `PROMPT_QA_RAG` | 复杂问答，需要跨时间段的语义检索 |

### 优势

1. **更精准的检索**：RAG基于语义相似度检索，不局限于时间戳附近
2. **多模态融合**：同时利用视觉和文本信息，提供更全面的理解
3. **时间戳关联**：检索结果包含时间戳，便于引用和验证
4. **上下文增强**：结合局部转录、全局摘要和检索片段，提供丰富的上下文

---

## 双模式部署架构

### 概述

项目支持 **Ollama** 和 **vLLM** 两种 LLM 部署模式，可根据环境、需求和资源情况灵活选择。两种模式在代码中并存，通过注释/取消注释即可切换。

### Ollama 模式（默认）

#### 架构设计

Ollama 模式使用本地 Ollama 服务进行推理，通过 REST API 调用。支持完整的多模态输入（图像 + 文本），模型以 Ollama 格式（GGUF/GGML）存储。

#### 实现细节

**1. 模型初始化**：

```python
# app/llm_utils.py
import ollama

def check_ollama_connection():
    """检查Ollama连接"""
    ollama.list()  # 测试连接
```

**2. 多模态消息构建**：

```python
# 构建包含图像的消息
messages = [{
    "role": "user",
    "content": prompt,  # 文本提示词
    "images": [base64_str1, base64_str2, ...]  # Base64编码的图像列表
}]

# 调用 Ollama Chat API
response = ollama.chat(
    model="qwen2.5-vl:7b",
    messages=messages,
    stream=True  # 流式输出
)
```

**3. 错误处理与降级**：

```python
try:
    response = ollama.chat(...)
except Exception:
    # 降级到 generate 方法
    result = ollama.generate(
        model=MODEL,
        prompt=prompt,
        images=prompt_inputs
    )
```

#### 配置方法

**1. 安装 Ollama 服务**：

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: 访问 https://ollama.com 下载安装包
```

**2. 下载模型**：

```bash
# 多模态模型（推荐）
ollama pull qwen2.5-vl:7b

# 或纯文本模型（更快）
ollama pull qwen3:8b
```

**3. 配置模型名称**（`app/config.py`）：

```python
MODEL = "qwen2.5-vl:7b"  # 或 "qwen3:8b"
```

**4. 启动服务**：

```bash
ollama serve
```

#### 优势与限制

**优势**：
- ✅ 安装简单，跨平台支持（macOS/Linux/Windows）
- ✅ 完整的多模态支持（图像理解）
- ✅ 模型管理便捷（`ollama pull` 即可）
- ✅ CPU 也可运行（速度较慢）
- ✅ 支持流式输出

**限制**：
- ⚠️ 推理速度中等（相比 vLLM）
- ⚠️ 并发处理能力有限
- ⚠️ 显存效率一般

#### 适用场景

- 开发测试环境
- 本地部署（无 GPU 或 GPU 资源有限）
- macOS 用户
- 需要完整多模态功能
- 快速原型开发

### vLLM 模式（高性能）

#### 架构设计

vLLM 模式使用 vLLM 高性能推理引擎，支持 Flash Attention 加速和 PagedAttention 显存优化。模型以 HuggingFace 格式存储，通过 LangChain 集成。

#### 实现细节

**1. 模型初始化**：

```python
# app/llm_utils.py
from langchain_community.llms import VLLM

@st.cache_resource
def get_vllm_model():
    """获取并缓存 vLLM 模型实例"""
    llm = VLLM(
        model=VLLM_MODEL_PATH,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.80,
        dtype="half"  # FP16
    )
    return llm
```

**2. 推理调用**：

```python
llm = get_vllm_model()
response = llm.generate([prompt])
```

**3. 多模态处理**：

vLLM 的多模态支持取决于模型：
- 如果模型支持（如 Qwen2.5-VL），可以直接处理图像
- 如果不支持，系统自动回退到文本模式，图像信息通过文本描述传递

#### 配置方法

**1. 安装依赖**：

```bash
pip install vllm>=0.6.0
```

**2. 配置模型路径**（`app/config.py`）：

```python
# 使用 HuggingFace 模型 ID（推荐）
VLLM_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# 或使用本地路径
VLLM_MODEL_PATH = "/path/to/local/model"

# vLLM 配置
VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.80,
    "tensor_parallel_size": 1,
    "dtype": "half",  # FP16
}
```

**3. 切换代码**（`app/llm_utils.py`）：
- 注释 Ollama 相关代码
- 取消注释 vLLM 相关代码

#### 优势与限制

**优势**：
- ✅ 推理速度快（Flash Attention 加速）
- ✅ 显存效率高（PagedAttention）
- ✅ 支持高并发请求（连续批处理）
- ✅ 适合生产环境
- ✅ 支持量化（int8/int4）

**限制**：
- ⚠️ 需要 NVIDIA GPU 和 CUDA
- ⚠️ 仅支持 Linux 环境
- ⚠️ 多模态支持有限（部分模型）
- ⚠️ 安装配置较复杂

#### 适用场景

- 生产环境部署
- 高并发需求
- 有充足 GPU 资源
- 追求最佳性能
- Linux 服务器环境

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
| **模型管理** | `ollama pull` | HuggingFace 下载 |
| **流式输出** | ✅ 支持 | ✅ 支持 |

### 模式切换

#### 从 Ollama 切换到 vLLM

**步骤 1**：修改 `app/config.py`

```python
# 注释 Ollama 配置
# MODEL = "qwen3:8b"

# 取消注释 vLLM 配置
VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
VLLM_CONFIG = {
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.80,
    "tensor_parallel_size": 1,
    "dtype": "half",
}
```

**步骤 2**：修改 `app/llm_utils.py`

```python
# 注释 Ollama 导入（第 7-17 行）
# import ollama
# OLLAMA_AVAILABLE = True

# 取消注释 vLLM 导入（第 19-28 行）
from langchain_community.llms import VLLM
from vllm import LLM, SamplingParams
VLLM_AVAILABLE = True

# 注释 Ollama 函数（第 130-155 行）
# def check_ollama_connection(): ...

# 取消注释 vLLM 函数（第 157-305 行）
# @st.cache_resource
# def get_vllm_model(): ...
```

**步骤 3**：更新 `get_response()` 函数

将函数中的 Ollama 调用替换为 vLLM 调用。

#### 从 vLLM 切换回 Ollama

反向操作上述步骤，并确保 Ollama 服务正在运行：

```bash
ollama serve
```

### 模型对应关系

| Ollama 模型 | HuggingFace 模型 | 类型 | 说明 |
|------------|-----------------|------|------|
| `qwen3:8b` | `Qwen/Qwen2.5-8B-Instruct` | 纯文本 | 文本理解优秀 |
| `qwen2.5-vl:7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | 多模态 | 支持图像理解 |

**注意**：
- vLLM 需要使用 HuggingFace 格式，不能直接使用 Ollama 格式
- 如果已通过 Ollama 下载模型，需要重新从 HuggingFace 下载对应模型

### 性能对比

**推理速度**（RTX 4090D 24GB）：
- **Ollama**: 20-40 tokens/秒
- **vLLM**: 50-100 tokens/秒

**显存占用**（7B 模型）：
- **Ollama**: ~16-20GB
- **vLLM**: ~18-22GB（但效率更高）

**并发能力**：
- **Ollama**: 有限（单请求处理）
- **vLLM**: 优秀（连续批处理，支持多请求）

---

## 技术细节

### 多模态融合

项目支持两种 LLM 部署模式，可根据环境和需求灵活选择：

#### Ollama 模式（默认）

**架构特点**：
- 使用 Ollama 本地服务进行推理
- 支持完整的多模态输入（图像 + 文本）
- 通过 REST API 调用，支持流式输出
- 模型格式：Ollama 格式（GGUF/GGML）

**实现方式**：
```python
# app/llm_utils.py
import ollama

# 构建多模态消息
messages = [{
    "role": "user",
    "content": prompt,
    "images": [base64_image1, base64_image2, ...]  # Base64编码的图像
}]

# 调用 Ollama API
response = ollama.chat(
    model="qwen2.5-vl:7b",
    messages=messages,
    stream=True
)
```

**优势**：
- ✅ 安装简单，跨平台支持
- ✅ 完整的多模态支持（图像理解）
- ✅ 模型管理便捷（`ollama pull`）
- ✅ CPU 也可运行（速度较慢）

**适用场景**：
- 开发测试环境
- 本地部署（无 GPU 或 GPU 资源有限）
- macOS 用户
- 需要完整多模态功能

#### vLLM 模式（高性能）

**架构特点**：
- 使用 vLLM 高性能推理引擎
- 支持 Flash Attention 加速
- 使用 PagedAttention 优化显存
- 支持连续批处理，并发能力强
- 模型格式：HuggingFace 格式

**实现方式**：
```python
# app/llm_utils.py
from langchain_community.llms import VLLM

# 初始化 vLLM 模型
llm = VLLM(
    model=VLLM_MODEL_PATH,
    trust_remote_code=True,
    max_model_len=8192,
    gpu_memory_utilization=0.80,
    dtype="half"
)

# 生成响应
response = llm.generate([prompt])
```

**优势**：
- ✅ 推理速度快（Flash Attention）
- ✅ 显存效率高（PagedAttention）
- ✅ 支持高并发请求
- ✅ 适合生产环境

**限制**：
- ⚠️ 需要 NVIDIA GPU 和 CUDA
- ⚠️ 仅支持 Linux 环境
- ⚠️ 多模态支持有限（部分模型）

**适用场景**：
- 生产环境部署
- 高并发需求
- 有充足 GPU 资源
- 追求最佳性能

#### 模式切换机制

**代码结构**：
- `app/config.py`: 配置模型路径和参数
- `app/llm_utils.py`: 包含两种模式的实现代码
  - Ollama 代码：默认启用
  - vLLM 代码：注释状态，需要时取消注释

**切换步骤**：
1. 修改 `app/config.py` 中的配置
2. 在 `app/llm_utils.py` 中注释/取消注释对应代码
3. 更新 `get_response()` 函数调用

**模型对应关系**：

| Ollama 模型 | HuggingFace 模型 | 说明 |
|------------|-----------------|------|
| `qwen3:8b` | `Qwen/Qwen2.5-8B-Instruct` | 纯文本模型 |
| `qwen2.5-vl:7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | 多模态模型 |

**注意**：vLLM 需要使用 HuggingFace 格式，不能直接使用 Ollama 格式。

### 多模态融合

系统结合多种信息源，实现真正的多模态理解：

1. **音频转录**: Whisper 生成的文本（时间戳对齐）
2. **视觉帧**: OpenCV 提取的视频帧（Base64编码）
3. **语义检索**: RAG 检索的相关片段（带时间戳）
4. **全局摘要**: 完整视频的文本摘要

**融合方式**：

**Ollama 模式**：
- 在提示词层面明确告知模型会收到图像和文本
- 通过 Ollama API 同时传递图像（Base64）和文本（Prompt）
- 多模态 LLM 在模型内部进行视觉-文本融合处理

**vLLM 模式**：
- 多模态支持取决于模型（如 Qwen2.5-VL 支持）
- 如果模型不支持，系统会回退到文本模式
- 图像信息通过文本提示词描述传递

### 提示词工程

不同任务使用不同的提示词模板，每个模板都明确说明了会提供的多模态信息：

- `video_qa`: 基础问答（图像 + 转录文本）
- `video_qa_rag`: RAG增强问答（图像 + 转录文本 + RAG检索文本）
- `video_summary`: 完整视频总结（关键帧图像 + 完整转录）
- `bullet_points`: 要点提取（仅文本）
- `qa_style`: 问答对生成（仅文本）

**提示词设计原则**：
1. 明确告知模型会收到哪些类型的信息（图像、文本等）
2. 要求模型同时使用所有提供的信息
3. 强调基于视频内容回答，避免生成推广内容
4. RAG模式下明确要求引用时间戳信息

**双模式提示词差异**：

- **Ollama 模式**：提示词中明确说明会提供图像，模型可以直接处理
- **vLLM 模式**：如果模型不支持多模态，提示词会包含图像描述文本

### 错误处理

**Ollama 模式**：
- 自动重试机制（Ollama 连接）
- 降级方案（stream 失败时使用 generate）
- 连接检查（`check_ollama_connection()`）
- 友好的错误提示
- 日志记录

**vLLM 模式**：
- CUDA 可用性检查
- 显存不足时的友好提示
- 模型加载失败时的详细错误信息
- 多模态输入失败时自动回退到文本模式
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

### Q: 多模态处理需要什么模型？

**Ollama 模式**：
- 推荐使用 `qwen2.5-vl:7b`（完整多模态支持）
- 或 `qwen3:8b`（纯文本，但可通过文本描述理解图像）
- 确保 Ollama 服务正在运行：`ollama serve`
- 下载模型：`ollama pull qwen2.5-vl:7b`

**vLLM 模式**：
- 推荐使用 `Qwen/Qwen2.5-VL-7B-Instruct`（多模态模型）
- 需要 NVIDIA GPU 和 CUDA 支持
- 模型会自动从 HuggingFace 下载
- 确保显存充足（建议 24GB+）

### Q: RAG检索和图像处理是同时进行的吗？

A: 是的。在RAG增强模式下：
1. 首先进行RAG文本检索（基于语义相似度）
2. 同时提取时间戳附近的视频帧
3. 将RAG检索的文本嵌入到提示词中
4. **Ollama 模式**：将图像通过API的`images`字段传递
5. **vLLM 模式**：如果模型支持多模态，直接传递；否则通过文本描述传递
6. 模型同时接收并融合处理图像和文本信息

### Q: 如何选择 Ollama 还是 vLLM 模式？

A: 选择建议：

**选择 Ollama 如果**：
- ✅ 开发测试环境
- ✅ 本地部署（无 GPU 或 GPU 资源有限）
- ✅ macOS 用户
- ✅ 需要完整的多模态支持（图像理解）
- ✅ 希望简单易用

**选择 vLLM 如果**：
- ✅ 生产环境部署
- ✅ 有 NVIDIA GPU 和 CUDA
- ✅ 需要高性能推理
- ✅ 需要支持高并发
- ✅ Linux 环境

### Q: 如何在两种模式间切换？

A: 切换步骤：

1. **切换到 vLLM**：
   - 在 `app/llm_utils.py` 中注释 Ollama 代码，取消注释 vLLM 代码
   - 在 `app/config.py` 中配置 `VLLM_MODEL_PATH` 和 `VLLM_CONFIG`
   - 确保已安装 vLLM：`pip install vllm>=0.6.0`

2. **切换回 Ollama**：
   - 在 `app/llm_utils.py` 中取消注释 Ollama 代码，注释 vLLM 代码
   - 在 `app/config.py` 中配置 `MODEL` 变量
   - 确保 Ollama 服务运行：`ollama serve`

详细说明请参考 [README.md](README.md) 的"双模式部署"部分。

### Q: 为什么需要Base64编码图像？

A: Ollama API要求图像以Base64编码的字符串形式传递，这是多模态API的标准格式。Base64编码可以将二进制图像数据转换为文本格式，便于在JSON消息中传输。

---

*最后更新: 2025-12-14*

