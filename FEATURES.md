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
- **混合检索**: 结合关键词检索 (`Keyword Search`) 补充语义检索的不足
- **多样性排序**: 支持 MMR (Maximal Marginal Relevance) 算法，减少重复内容
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
查询时：
  1. 向量检索 (Vector Search)
  2. 关键词检索 (Keyword Search) [可选]
  3. MMR 重排序 [可选]
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

---

## 多模态视觉处理

### 概述

本项目充分利用多模态技术，将视频的视觉信息（视频帧）与文本信息（转录文本）相结合，通过支持视觉的多模态 LLM 进行综合分析和问答（建议使用支持视觉的模型，例如 `qwen2.5-vl:7b`）。

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
    model="qwen2.5-vl:7b",  # 支持视觉的多模态模型
    messages=messages,      # 包含图像的消息列表
    stream=True             # 流式输出
)
```

**3. 降级方案**

如果流式 `ollama.chat` 失败，先使用非流式 `ollama.chat`，仍失败再使用 `ollama.generate` 作为降级：

```python
result = ollama.generate(
    model="qwen2.5-vl:7b",
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
    model="qwen2.5-vl:7b",
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

#### 5. 模型视觉支持说明

**当前状态**：

项目默认 `MODEL` 配置在 `app/config.py` 中（当前为 `qwen3:8b`）。注意：

1. **默认模型可能不支持视觉**：
   - `qwen3:8b` 为纯文本模型，不支持视觉输入
   - 传递图像会被忽略或导致错误（取决于模型实现）

2. **如果使用非视觉模型**：
   - 模型只会基于文本 prompt 生成回答
   - 视觉信息不会被利用

3. **建议的视觉模型**：
   - **Qwen VL 系列**：`qwen2.5-vl:7b`
   - **llava 系列**：`llava:7b`, `llava:13b`, `llava:34b`
   - **其他视觉模型**：如`bakllava`, `minicpm-v`等

**如何验证模型是否支持视觉**：

```bash
# 检查模型信息
ollama show qwen3:8b

# 尝试传递图像测试
ollama run qwen2.5-vl:7b "描述这张图片" --images image.jpg
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
PROMPT_QA_RAG = """你是一位专业的教学助手。你将获得以下信息：
- 一些视频帧（图像）
- 特定时间戳附近的转录片段
- 通过语义检索找到的相关视频片段（带时间戳）
- 视频摘要作为背景信息

你的任务是基于所有提供的信息回答问题。当引用特定内容时，请提及相应的时间戳（如果可用）。

重要提示：请勿在回答中包含任何推广内容、订阅链接、新闻通讯提及、网站URL或广告信息。只基于视频内容回答问题。

请按照以下思维链步骤进行深入分析：

**第一步：理解问题**
- 仔细分析用户的问题：{question}
- 确定问题的核心和所需的信息类型
- 识别问题可能涉及的关键概念

**第二步：分析检索到的相关片段**
语义检索到的相关片段（按相关性排序）：
{retrieved_contexts}

- 分析每个相关片段与问题的关联度
- 提取每个片段中的关键信息
- 注意时间戳，理解内容的时间顺序

**第三步：整合全局上下文**
全局视频摘要：
{global_context}

- 理解视频的整体主题和结构
- 将检索到的片段与全局上下文关联
- 识别信息之间的逻辑关系

**第四步：分析聚焦片段**
时间戳附近的聚焦转录片段：
{text}

- 提取与问题直接相关的细节
- 分析片段在整体视频中的位置和作用
- 识别关键概念和具体信息

**第五步：综合推理**
- 将检索片段、全局上下文和聚焦片段进行综合
- 建立信息之间的逻辑链条
- 识别可能的信息缺口或需要进一步分析的部分
- 评估不同信息源的可信度和相关性

**第六步：构建答案**
- 基于综合推理结果，构建完整、准确的答案
- 按照逻辑顺序组织信息
- 引用具体的时间戳和内容片段
- 确保答案直接回应问题的所有方面

**最终答案：**
基于以上完整的思维链分析，请给出详细、准确且结构化的回答："""
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
    "content": """你是一位专业的教学助手。你将获得以下信息：
- 一些视频帧（图像）
- 通过语义检索找到的相关视频片段（带时间戳）
...

语义检索到的相关片段（按相关性排序）：
[时间戳: 1:30] RAG检索到的第一段文本...
[时间戳: 2:15] RAG检索到的第二段文本...

全局视频摘要：
视频的完整摘要...

时间戳附近的聚焦转录片段：
时间戳附近的转录片段...

问题：用户的问题
回答：""",
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
## 3. 部署与配置 (Deployment & Configuration)

详细的部署指南（包括 Ollama 和 vLLM 模式的安装、配置与对比）请参阅独立文档：

👉 **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

---

## 技术细节

### 多模态融合（信息源）

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
- 多模态支持取决于模型与实现（如 Qwen2.5-VL 支持）
- 当前代码未实现图像传递，默认文本模式

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
- 降级方案（stream 失败先用非流式 chat，再 fallback 到 generate）
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
5. **vLLM 模式**：当前未实现图像传递，默认仅文本推理
6. 模型同时接收并融合处理图像和文本信息

### Q: 如何选择 Ollama 还是 vLLM 模式？

A: 参考“模式对比”表中的“优势/限制/适用场景（汇总）”行进行选择。

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
