from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import (
    PathConstants,
    RAGConstants,
    default_logger,
)


class VideoRAGSystem:
    """基于LangChain的视频RAG系统"""
    
    def __init__(self, embedding_model: Embeddings):
        """
        初始化RAG系统
        :param embedding_model: 用于向量化的模型实例
        """
        self.embedding_model = embedding_model
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConstants.DEFAULT_CHUNK_SIZE,
            chunk_overlap=RAGConstants.DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        self.index_path = PathConstants.FAISS_INDEX_DIR
        self.logger = default_logger
    
    def create_documents_from_segments(self, segments: List[Dict]) -> List[Document]:
        """
        将Whisper转录片段转换为LangChain Document对象
        保留时间戳元数据
        """
        documents = []
        for seg in segments:
            # 创建包含时间戳信息的文档
            doc = Document(
                page_content=seg['text'],
                metadata={
                    'start_time': seg['start'],
                    'end_time': seg['end'],
                    'timestamp': f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
                }
            )
            documents.append(doc)
        return documents
    
    def build_vector_store(
        self,
        segments: List[Dict],
        video_id: Optional[str] = None
    ) -> None:
        """
        构建向量存储
        
        Args:
            segments: Whisper转录片段
            video_id: 视频ID，用于多视频场景
        
        Raises:
            ValueError: 如果segments为空
        """
        if not segments:
            raise ValueError("无法构建向量索引：转录片段为空")
        
        # 转换为Document对象
        documents = self.create_documents_from_segments(segments)
        
        # 按时间窗口合并相关片段（可选）
        # 如果片段太短，可以合并相邻片段
        merged_docs = self._merge_short_segments(documents)
        
        # 分割长文档
        split_docs = self.text_splitter.split_documents(merged_docs)
        
        if not split_docs:
            raise ValueError("无法构建向量索引：文档分割后为空")
        
        # 构建FAISS向量存储
        self.vectorstore = FAISS.from_documents(
            documents=split_docs,
            embedding=self.embedding_model
        )
        
        # 保存索引
        index_dir = self._get_index_path(video_id)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        self.vectorstore.save_local(str(index_dir))
        self.logger.info("向量索引已构建: %s (segments=%s)", index_dir, len(split_docs))
        
        # 验证保存是否成功
        faiss_file = index_dir / "index.faiss"
        pkl_file = index_dir / "index.pkl"
        if faiss_file.exists() and pkl_file.exists():
            st.success(f"✅ 向量索引已构建并保存到 {index_dir}")
            self.logger.info("向量索引已验证: %s", index_dir)
        else:
            st.warning(f"⚠️ 向量索引构建完成，但保存验证失败: {index_dir}")
            self.logger.warning("向量索引构建完成，但保存验证失败: %s", index_dir)
    
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
    
    def _get_index_path(self, video_id: Optional[str] = None) -> Path:
        """获取索引目录路径"""
        if video_id:
            return Path(f"{self.index_path}_{video_id}")
        return Path(self.index_path)

    def _is_index_valid(self, index_dir: Path) -> bool:
        """检查索引目录是否有效"""
        if not index_dir.exists() or not index_dir.is_dir():
            return False
        faiss_file = index_dir / "index.faiss"
        pkl_file = index_dir / "index.pkl"
        return faiss_file.exists() and pkl_file.exists()

    def load_vector_store(self, video_id: Optional[str] = None) -> bool:
        """
        加载已保存的向量存储
        
        Args:
            video_id: 视频ID，用于多视频场景
        
        Returns:
            bool: 是否成功加载
        """
        index_dir = self._get_index_path(video_id)
        index_path = str(index_dir)
        
        if not self._is_index_valid(index_dir):
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            self.logger.info("成功加载向量索引: %s", index_path)
            return True
        except FileNotFoundError:
            # 文件不存在，这是正常情况，不需要警告
            return False
        except Exception as e:
            # 其他错误（如文件损坏），记录警告
            st.warning(f"⚠️ 无法加载向量索引: {e}")
            self.logger.warning("无法加载向量索引: %s", e, exc_info=True)
            return False
    
    def retrieve_relevant_context(
        self,
        query: str,
        top_k: int = RAGConstants.DEFAULT_TOP_K,
        score_threshold: float = RAGConstants.DEFAULT_SCORE_THRESHOLD,
        use_mmr: bool = False,
        fetch_k: int = 15,
        lambda_mult: float = 0.5,
        use_hybrid: bool = False
    ) -> List[Dict[str, Any]]:
        """
        检索相关上下文
        :param query: 查询问题
        :param top_k: 返回top-k个相关片段
        :param score_threshold: 相似度阈值（L2距离，越小越相似）
        :param use_mmr: 是否启用 MMR 检索以提升多样性
        :param fetch_k: MMR 预检索数量
        :param lambda_mult: MMR 多样性参数
        :param use_hybrid: 是否混合语义+关键词检索
        :return: 相关上下文列表，包含文本、时间戳等信息
        """
        if self.vectorstore is None:
            return []
        
        docs_with_scores: Sequence[Any] = []

        if use_mmr:
            try:
                mmr_docs = self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=top_k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                )
                docs_with_scores = [(doc, 0.0) for doc in mmr_docs]
            except Exception as e:
                self.logger.warning("MMR 检索失败，回退普通检索: %s", e)

        if not docs_with_scores:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )

        keyword_docs: List[Document] = []
        if use_hybrid:
            keyword_docs = self._keyword_search(query, top_k=top_k)
        
        # 过滤高分结果（L2距离，越小越好）并格式化
        contexts = []
        for doc, score in docs_with_scores:
            similarity_score = 1 / (1 + score) if score > 0 else 1.0
            if score <= score_threshold or use_mmr:
                contexts.append(self._doc_to_context(doc, similarity_score))

        if keyword_docs:
            keyword_contexts = [
                self._doc_to_context(doc, score=0.0) for doc in keyword_docs
            ]
            contexts.extend(keyword_contexts)

        # 去重并按分数排序
        return self._dedup_contexts(contexts)[:top_k]
    
    def retrieve_around_timestamp(
        self,
        timestamp: float,
        window: int = 5,
        query: Optional[str] = None,
        top_k: int = RAGConstants.DEFAULT_TOP_K
    ) -> Dict[str, Any]:
        """
        结合时间戳和语义检索
        :param timestamp: 目标时间戳
        :param window: 时间窗口（秒）
        :param query: 可选的问题，用于语义过滤
        :return: 相关上下文
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
                top_k=max(top_k, 5),
                use_mmr=True,
                use_hybrid=True
            )
            # 过滤出时间戳附近的片段
            time_filtered = [
                ctx for ctx in semantic_results
                if abs(ctx['start_time'] - timestamp) <= window * 2
            ]
            if time_filtered:
                return {
                    'text': ' '.join([ctx['text'] for ctx in time_filtered]),
                    'contexts': time_filtered,
                    'retrieval_type': 'semantic_temporal',
                    'timestamp': timestamp,
                    'window': window
                }

        # 时间窗口检索：从 docstore 过滤靠近时间戳的文档
        nearby_contexts = self._retrieve_by_time_window(timestamp, window, limit=top_k)
        if nearby_contexts:
            return {
                'text': ' '.join([ctx['text'] for ctx in nearby_contexts]),
                'contexts': nearby_contexts,
                'retrieval_type': 'temporal_only',
                'timestamp': timestamp,
                'window': window
            }

        # 回退：返回空结果，但标记原因
        return {
            'text': '',
            'contexts': [],
            'retrieval_type': 'temporal_only',
            'note': '未找到时间窗口内的片段'
        }
    
    def cleanup_invalid_indices(self, keep_signatures: Optional[List[str]] = None):
        """
        清理无效或过期的索引文件
        
        Args:
            keep_signatures: 要保留的视频签名列表（可选）
        """
        base_path = Path(self.index_path).parent
        pattern = "faiss_index*"
        
        cleaned_count = 0
        for index_dir in base_path.glob(pattern):
            if not index_dir.is_dir():
                continue
            
            # 检查索引文件是否完整
            if not self._is_index_valid(index_dir):
                try:
                    import shutil
                    shutil.rmtree(index_dir)
                    cleaned_count += 1
                    self.logger.info("已删除无效索引: %s", index_dir)
                except Exception as e:
                    st.warning(f"⚠️ 无法删除无效索引 {index_dir}: {e}")
                    self.logger.warning("无法删除无效索引 %s: %s", index_dir, e, exc_info=True)
            
            # 如果指定了要保留的签名，检查是否应该删除
            elif keep_signatures:
                dir_name = index_dir.name
                if dir_name.startswith("faiss_index_"):
                    signature = dir_name.replace("faiss_index_", "")
                    if signature not in keep_signatures:
                        try:
                            import shutil
                            shutil.rmtree(index_dir)
                            cleaned_count += 1
                            self.logger.info("已删除过期索引: %s", index_dir)
                        except Exception as e:
                            st.warning(f"⚠️ 无法删除过期索引 {index_dir}: {e}")
                            self.logger.warning("无法删除过期索引 %s: %s", index_dir, e, exc_info=True)
        
        if cleaned_count > 0:
            st.info(f"🧹 已清理 {cleaned_count} 个无效索引")
    
    def _doc_to_context(self, doc: Document, score: float) -> Dict[str, Any]:
        """将 Document 转换为上下文字典"""
        return {
            'text': doc.page_content,
            'start_time': doc.metadata.get('start_time', 0),
            'end_time': doc.metadata.get('end_time', 0),
            'timestamp': doc.metadata.get('timestamp', ''),
            'score': float(score)
        }

    def _dedup_contexts(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按时间戳和文本去重"""
        seen = set()
        unique_contexts = []
        for ctx in contexts:
            key = (ctx.get('start_time'), ctx.get('timestamp'), ctx.get('text'))
            if key in seen:
                continue
            seen.add(key)
            unique_contexts.append(ctx)
        return sorted(unique_contexts, key=lambda c: c.get('score', 0), reverse=True)

    def _keyword_search(self, query: str, top_k: int) -> List[Document]:
        """简单关键词检索，作为语义检索补充"""
        if self.vectorstore is None or not hasattr(self.vectorstore, "docstore"):
            return []
        tokens = {tok.lower() for tok in query.split() if tok}
        scored_docs = []
        try:
            for doc in self.vectorstore.docstore._dict.values():
                content = doc.page_content.lower()
                overlap = sum(1 for tok in tokens if tok in content)
                if overlap:
                    scored_docs.append((doc, overlap))
        except Exception as e:
            self.logger.debug("关键词检索失败: %s", e, exc_info=True)
            return []
        scored_docs.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    def _retrieve_by_time_window(
        self,
        timestamp: float,
        window: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """基于时间窗口的简单检索"""
        if self.vectorstore is None or not hasattr(self.vectorstore, "docstore"):
            return []
        candidates: List[Document] = []
        try:
            for doc in self.vectorstore.docstore._dict.values():
                start_time = doc.metadata.get('start_time', 0)
                if abs(start_time - timestamp) <= window:
                    candidates.append(doc)
        except Exception as e:
            self.logger.debug("时间窗口检索失败: %s", e, exc_info=True)
            return []
        candidates.sort(key=lambda d: d.metadata.get('start_time', 0))
        contexts = [self._doc_to_context(doc, score=0.0) for doc in candidates]
        return contexts[:limit]
