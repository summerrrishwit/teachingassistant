import os
from pathlib import Path
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
import streamlit as st
from config import VIDEO_PATH, FRAME_DIR


class VideoRAGSystem:
    """基于LangChain的视频RAG系统"""
    
    def __init__(self, embedding_model):
        """
        初始化RAG系统
        :param embedding_model: 用于向量化的模型实例
        """
        self.embedding_model = embedding_model
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 每个chunk的字符数
            chunk_overlap=50,  # chunk之间的重叠
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        self.index_path = "runtime/faiss_index"
    
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
    
    def build_vector_store(self, segments: List[Dict], video_id: str = None):
        """
        构建向量存储
        
        Args:
            segments: Whisper转录片段
            video_id: 视频ID，用于多视频场景
        
        Raises:
            ValueError: 如果segments为空
        """
        from pathlib import Path
        
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
        if video_id:
            index_path = f"{self.index_path}_{video_id}"
        else:
            index_path = self.index_path
        
        # 确保目录存在
        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        self.vectorstore.save_local(str(index_path))
        
        # 验证保存是否成功
        faiss_file = index_dir / "index.faiss"
        pkl_file = index_dir / "index.pkl"
        if faiss_file.exists() and pkl_file.exists():
            st.success(f"✅ 向量索引已构建并保存到 {index_path}")
        else:
            st.warning(f"⚠️ 向量索引构建完成，但保存验证失败: {index_path}")
    
    def _merge_short_segments(self, documents: List[Document], min_length: int = 50) -> List[Document]:
        """合并过短的片段"""
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
    
    def load_vector_store(self, video_id: str = None):
        """
        加载已保存的向量存储
        
        Args:
            video_id: 视频ID，用于多视频场景
        
        Returns:
            bool: 是否成功加载
        """
        from pathlib import Path
        
        # 确定索引路径
        if video_id:
            index_path = f"{self.index_path}_{video_id}"
        else:
            index_path = self.index_path
        
        # 检查索引文件是否存在
        index_dir = Path(index_path)
        faiss_file = index_dir / "index.faiss"
        pkl_file = index_dir / "index.pkl"
        
        if not index_dir.exists():
            return False
        
        if not faiss_file.exists() or not pkl_file.exists():
            # 索引文件不完整，返回False以便重新构建
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            return True
        except FileNotFoundError:
            # 文件不存在，这是正常情况，不需要警告
            return False
        except Exception as e:
            # 其他错误（如文件损坏），记录警告
            st.warning(f"⚠️ 无法加载向量索引: {e}")
            return False
    
    def retrieve_relevant_context(
        self, 
        query: str, 
        top_k: int = 3,
        score_threshold: float = 1.5  # FAISS使用的是L2距离，值越大表示越不相似
    ) -> List[Dict]:
        """
        检索相关上下文
        :param query: 查询问题
        :param top_k: 返回top-k个相关片段
        :param score_threshold: 相似度阈值（L2距离，越小越相似）
        :return: 相关上下文列表，包含文本、时间戳等信息
        """
        if self.vectorstore is None:
            return []
        
        # 使用相似度搜索
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, 
            k=top_k
        )
        
        # 过滤高分结果（L2距离，越小越好）并格式化
        contexts = []
        for doc, score in docs_with_scores:
            # 转换L2距离为相似度分数（0-1之间，越大越相似）
            similarity_score = 1 / (1 + score) if score > 0 else 1.0
            if score <= score_threshold:  # L2距离阈值
                contexts.append({
                    'text': doc.page_content,
                    'start_time': doc.metadata.get('start_time', 0),
                    'end_time': doc.metadata.get('end_time', 0),
                    'timestamp': doc.metadata.get('timestamp', ''),
                    'score': float(similarity_score)  # 相似度分数
                })
        
        return contexts
    
    def retrieve_around_timestamp(
        self, 
        timestamp: float, 
        window: int = 5,
        query: str = None
    ) -> Dict:
        """
        结合时间戳和语义检索
        :param timestamp: 目标时间戳
        :param window: 时间窗口（秒）
        :param query: 可选的问题，用于语义过滤
        :return: 相关上下文
        """
        if self.vectorstore is None:
            return {}
        
        # 如果提供了查询，先进行语义检索
        if query:
            semantic_results = self.retrieve_relevant_context(query, top_k=5)
            # 过滤出时间戳附近的片段
            time_filtered = [
                ctx for ctx in semantic_results
                if abs(ctx['start_time'] - timestamp) <= window * 2
            ]
            if time_filtered:
                return {
                    'text': ' '.join([ctx['text'] for ctx in time_filtered]),
                    'contexts': time_filtered,
                    'retrieval_type': 'semantic_temporal'
                }
        
        # 否则，直接从向量存储中查找时间戳附近的文档
        # 这里需要遍历所有文档（可以优化为基于时间戳的索引）
        # 简化实现：返回空，让上层使用 transcript_utils 的功能
        return {
            'text': '',
            'contexts': [],
            'retrieval_type': 'temporal_only'
        }
    
    def cleanup_invalid_indices(self, keep_signatures: List[str] = None):
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
            faiss_file = index_dir / "index.faiss"
            pkl_file = index_dir / "index.pkl"
            
            # 如果文件不完整，删除目录
            if not faiss_file.exists() or not pkl_file.exists():
                try:
                    import shutil
                    shutil.rmtree(index_dir)
                    cleaned_count += 1
                except Exception as e:
                    st.warning(f"⚠️ 无法删除无效索引 {index_dir}: {e}")
            
            # 如果指定了要保留的签名，检查是否应该删除
            elif keep_signatures:
                # 从目录名提取签名
                dir_name = index_dir.name
                if dir_name.startswith("faiss_index_"):
                    signature = dir_name.replace("faiss_index_", "")
                    if signature not in keep_signatures:
                        try:
                            import shutil
                            shutil.rmtree(index_dir)
                            cleaned_count += 1
                        except Exception as e:
                            st.warning(f"⚠️ 无法删除过期索引 {index_dir}: {e}")
        
        if cleaned_count > 0:
            st.info(f"🧹 已清理 {cleaned_count} 个无效索引")

