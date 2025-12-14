# 🔧 rag_utils.py 优化建议

本文档详细列出了 `rag_utils.py` 中可以优化的地方。

## 📋 优化清单

### 🔴 高优先级（立即优化）

#### 1. **使用常量替代硬编码值**

**问题**：
- 第22-26行：`chunk_size=500`, `chunk_overlap=50` 硬编码
- 第28行：`index_path = "runtime/faiss_index"` 硬编码
- 第105行：`min_length: int = 50` 硬编码
- 第173行：`score_threshold: float = 1.5` 硬编码

**建议**：
```python
from core import RAGConstants, PathConstants

class VideoRAGSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConstants.DEFAULT_CHUNK_SIZE,
            chunk_overlap=RAGConstants.DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        self.index_path = str(PathConstants.FAISS_INDEX_DIR)
```

#### 2. **改进导入和依赖管理**

**问题**：
- 第9行：导入了未使用的 `VIDEO_PATH, FRAME_DIR`
- 第60行、134行：在函数内部导入 `Path`（应该在顶部导入）
- 第269行、283行：在函数内部导入 `shutil`（应该在顶部导入）

**建议**：
```python
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import streamlit as st
from core import RAGConstants, PathConstants
```

#### 3. **添加类型提示**

**问题**：
- 第15行：`embedding_model` 缺少类型提示
- 第49行：`video_id: str = None` 应该使用 `Optional[str]`
- 第124行：同样的问题
- 第247行：`keep_signatures: List[str] = None` 应该使用 `Optional[List[str]]`

**建议**：
```python
from typing import List, Dict, Optional, Any
from langchain_core.embeddings import Embeddings

class VideoRAGSystem:
    def __init__(self, embedding_model: Embeddings):
        ...
    
    def build_vector_store(self, segments: List[Dict], video_id: Optional[str] = None) -> None:
        ...
    
    def load_vector_store(self, video_id: Optional[str] = None) -> bool:
        ...
    
    def cleanup_invalid_indices(self, keep_signatures: Optional[List[str]] = None) -> None:
        ...
```

#### 4. **添加日志记录**

**问题**：
- 缺少日志记录，难以调试和追踪问题
- 错误信息只通过 `st.warning` 显示，没有记录到日志

**建议**：
```python
from core import setup_logger

logger = setup_logger(__name__)

class VideoRAGSystem:
    def build_vector_store(self, segments: List[Dict], video_id: Optional[str] = None) -> None:
        logger.info(f"开始构建向量索引，视频ID: {video_id}, 片段数量: {len(segments)}")
        try:
            # ... 构建逻辑
            logger.info(f"向量索引构建成功: {index_path}")
        except Exception as e:
            logger.error(f"向量索引构建失败: {e}", exc_info=True)
            raise
```

#### 5. **改进错误处理**

**问题**：
- 第166行：捕获所有异常，但只显示警告，没有记录详细错误
- 缺少具体的异常类型处理

**建议**：
```python
from core import RAGError, setup_logger

logger = setup_logger(__name__)

def load_vector_store(self, video_id: Optional[str] = None) -> bool:
    try:
        self.vectorstore = FAISS.load_local(...)
        logger.info(f"成功加载向量索引: {index_path}")
        return True
    except FileNotFoundError:
        logger.debug(f"索引文件不存在: {index_path}")
        return False
    except Exception as e:
        logger.error(f"加载向量索引失败: {e}", exc_info=True)
        raise RAGError(f"无法加载向量索引: {e}") from e
```

---

### 🟡 中优先级（近期优化）

#### 6. **提取路径处理逻辑**

**问题**：
- `_get_index_path()` 逻辑在多个方法中重复（第85-88行，第137-140行）

**建议**：
```python
def _get_index_path(self, video_id: Optional[str] = None) -> Path:
    """获取索引路径"""
    if video_id:
        return Path(f"{self.index_path}_{video_id}")
    return Path(self.index_path)

def build_vector_store(self, segments: List[Dict], video_id: Optional[str] = None) -> None:
    index_path = self._get_index_path(video_id)
    # ...
```

#### 7. **改进 `retrieve_around_timestamp` 方法**

**问题**：
- 第207-245行：方法实现不完整，注释说"可以优化为基于时间戳的索引"
- 第241行：直接返回空字典，功能不完整

**建议**：
```python
def retrieve_around_timestamp(
    self, 
    timestamp: float, 
    window: int = 5,
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    结合时间戳和语义检索
    
    Args:
        timestamp: 目标时间戳（秒）
        window: 时间窗口（秒）
        query: 可选的问题，用于语义过滤
    
    Returns:
        包含相关上下文的字典
    """
    if self.vectorstore is None:
        return {
            'text': '',
            'contexts': [],
            'retrieval_type': 'no_index'
        }
    
    # 如果提供了查询，先进行语义检索
    if query:
        semantic_results = self.retrieve_relevant_context(
            query, 
            top_k=10  # 增加检索数量以便过滤
        )
        # 过滤出时间戳附近的片段
        time_filtered = [
            ctx for ctx in semantic_results
            if abs(ctx['start_time'] - timestamp) <= window
        ]
        if time_filtered:
            return {
                'text': ' '.join([ctx['text'] for ctx in time_filtered]),
                'contexts': time_filtered,
                'retrieval_type': 'semantic_temporal',
                'timestamp': timestamp,
                'window': window
            }
    
    # 如果没有查询或过滤后为空，返回空结果
    # 注意：完整实现需要遍历所有文档或使用时间索引
    return {
        'text': '',
        'contexts': [],
        'retrieval_type': 'temporal_only',
        'note': '需要上层使用 transcript_utils 的功能'
    }
```

#### 8. **优化 `_merge_short_segments` 方法**

**问题**：
- 第105行：`min_length` 硬编码，应该使用常量
- 逻辑可以更清晰

**建议**：
```python
def _merge_short_segments(
    self, 
    documents: List[Document], 
    min_length: Optional[int] = None
) -> List[Document]:
    """
    合并过短的片段
    
    Args:
        documents: 文档列表
        min_length: 最小长度阈值，默认使用 RAGConstants.MIN_SEGMENT_LENGTH
    """
    if min_length is None:
        min_length = RAGConstants.MIN_SEGMENT_LENGTH
    
    merged = []
    current_doc = None
    
    for doc in documents:
        if len(doc.page_content) < min_length and current_doc:
            # 合并到当前文档
            current_doc.page_content += " " + doc.page_content
            current_doc.metadata['end_time'] = doc.metadata['end_time']
        else:
            if current_doc:
                merged.append(current_doc)
            current_doc = doc
    
    if current_doc:
        merged.append(current_doc)
    
    return merged
```

#### 9. **添加索引验证方法**

**问题**：
- 索引文件完整性检查逻辑在多个地方重复

**建议**：
```python
def _is_index_valid(self, index_dir: Path) -> bool:
    """
    检查索引目录是否有效
    
    Args:
        index_dir: 索引目录路径
    
    Returns:
        bool: 索引是否有效
    """
    if not index_dir.exists() or not index_dir.is_dir():
        return False
    
    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"
    
    return faiss_file.exists() and pkl_file.exists()
```

#### 10. **改进 `cleanup_invalid_indices` 方法**

**问题**：
- 第254行：`pattern = "faiss_index*"` 未使用
- 第269行、283行：重复导入 `shutil`
- 错误处理可以更细致

**建议**：
```python
def cleanup_invalid_indices(self, keep_signatures: Optional[List[str]] = None) -> int:
    """
    清理无效或过期的索引文件
    
    Args:
        keep_signatures: 要保留的视频签名列表（可选）
    
    Returns:
        int: 清理的索引数量
    """
    base_path = Path(self.index_path).parent
    cleaned_count = 0
    keep_set = set(keep_signatures) if keep_signatures else set()
    
    for index_dir in base_path.glob("faiss_index*"):
        if not index_dir.is_dir():
            continue
        
        # 检查索引文件是否完整
        if not self._is_index_valid(index_dir):
            try:
                shutil.rmtree(index_dir)
                cleaned_count += 1
                logger.info(f"已删除无效索引: {index_dir}")
            except Exception as e:
                logger.warning(f"无法删除无效索引 {index_dir}: {e}")
                continue
        
        # 如果指定了要保留的签名，检查是否应该删除
        if keep_signatures:
            dir_name = index_dir.name
            if dir_name.startswith("faiss_index_"):
                signature = dir_name.replace("faiss_index_", "")
                if signature not in keep_set:
                    try:
                        shutil.rmtree(index_dir)
                        cleaned_count += 1
                        logger.info(f"已删除过期索引: {index_dir}")
                    except Exception as e:
                        logger.warning(f"无法删除过期索引 {index_dir}: {e}")
    
    if cleaned_count > 0:
        st.info(f"🧹 已清理 {cleaned_count} 个无效索引")
        logger.info(f"清理完成，共清理 {cleaned_count} 个索引")
    
    return cleaned_count
```

---

### 🟢 低优先级（长期优化）

#### 11. **添加索引统计信息**

**建议**：
```python
def get_index_stats(self, video_id: Optional[str] = None) -> Dict[str, Any]:
    """
    获取索引统计信息
    
    Returns:
        包含索引统计信息的字典
    """
    if self.vectorstore is None:
        return {'status': 'no_index'}
    
    # 获取索引中的文档数量
    doc_count = len(self.vectorstore.docstore._dict) if hasattr(self.vectorstore, 'docstore') else 0
    
    return {
        'status': 'loaded',
        'document_count': doc_count,
        'index_path': str(self._get_index_path(video_id))
    }
```

#### 12. **支持索引增量更新**

**建议**：
```python
def add_documents(self, new_segments: List[Dict], video_id: Optional[str] = None) -> None:
    """
    增量添加文档到现有索引
    
    Args:
        new_segments: 新的转录片段
        video_id: 视频ID
    """
    if self.vectorstore is None:
        raise RAGError("向量索引未加载，无法添加文档")
    
    new_documents = self.create_documents_from_segments(new_segments)
    merged_docs = self._merge_short_segments(new_documents)
    split_docs = self.text_splitter.split_documents(merged_docs)
    
    # 添加到现有索引
    self.vectorstore.add_documents(split_docs)
    
    # 保存更新后的索引
    index_path = self._get_index_path(video_id)
    self.vectorstore.save_local(str(index_path))
    
    logger.info(f"已添加 {len(split_docs)} 个新文档到索引")
```

#### 13. **添加索引版本管理**

**建议**：
```python
def _get_index_version(self, index_dir: Path) -> Optional[str]:
    """获取索引版本号"""
    version_file = index_dir / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    return None

def _save_index_version(self, index_dir: Path, version: str = "1.0") -> None:
    """保存索引版本号"""
    version_file = index_dir / "version.txt"
    version_file.write_text(version)
```

#### 14. **优化检索性能**

**建议**：
- 使用 MMR (Maximal Marginal Relevance) 检索，提高结果多样性
- 支持混合检索（语义 + 关键词）
- 添加检索缓存

```python
def retrieve_with_mmr(
    self,
    query: str,
    top_k: int = 3,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> List[Dict]:
    """
    使用 MMR 检索，提高结果多样性
    
    Args:
        query: 查询问题
        top_k: 返回结果数量
        fetch_k: 初始检索数量
        lambda_mult: MMR 多样性参数 (0-1)
    """
    if self.vectorstore is None:
        return []
    
    docs = self.vectorstore.max_marginal_relevance_search(
        query,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )
    
    # 转换为标准格式
    return [self._doc_to_context(doc) for doc in docs]
```

---

## 📊 优化优先级总结

### 🔴 立即实施
1. ✅ 使用常量替代硬编码值
2. ✅ 改进导入和依赖管理
3. ✅ 添加类型提示
4. ✅ 添加日志记录
5. ✅ 改进错误处理

### 🟡 近期实施
6. 提取路径处理逻辑
7. 改进 `retrieve_around_timestamp` 方法
8. 优化 `_merge_short_segments` 方法
9. 添加索引验证方法
10. 改进 `cleanup_invalid_indices` 方法

### 🟢 长期规划
11. 添加索引统计信息
12. 支持索引增量更新
13. 添加索引版本管理
14. 优化检索性能

---

## 🎯 实施建议

1. **分阶段实施**：先处理高优先级项目
2. **保持向后兼容**：确保优化不影响现有功能
3. **添加测试**：每次优化后都要测试
4. **更新文档**：及时更新相关文档

---

## 📝 注意事项

- 所有优化都应该保持代码的可读性和可维护性
- 性能优化需要先进行基准测试
- 大型重构应该分步骤进行
- 确保所有导入路径正确

---

*最后更新: 2025-12-14*
