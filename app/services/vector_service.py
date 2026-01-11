"""
向量索引服务
处理向量索引的构建、加载和管理
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from app.llm_utils import get_rag_system


def _video_signature(video_path: Path) -> Optional[str]:
    """Generate a lightweight signature for the current video (size + mtime)."""
    try:
        stat = os.stat(video_path)
        payload = f"{video_path}:{stat.st_size}:{int(stat.st_mtime)}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    except OSError:
        return None


def clear_vector_index_state():
    """Clear cached vector index info when a new video is uploaded or state is reset."""
    rag_system = get_rag_system()
    rag_system.vectorstore = None
    for key in ["vector_index_built", "vector_index_signature"]:
        if key in st.session_state:
            del st.session_state[key]


def ensure_vector_index(segments: List[Dict], video_path: Path) -> bool:
    """
    Ensure vector index is available for the current video.
    Attempts to reuse cached signature and on-disk index; builds if missing.
    
    Args:
        segments: Whisper转录片段
        video_path: 视频文件路径
    
    Returns:
        bool: 是否成功确保索引可用
    """
    rag_system = get_rag_system()
    signature = _video_signature(video_path)

    # Reuse in-memory vectorstore if signature matches
    if (
        signature
        and st.session_state.get("vector_index_signature") == signature
        and rag_system.vectorstore is not None
    ):
        st.session_state.vector_index_built = True
        return True

    # Try loading from disk if exists
    if signature:
        if rag_system.load_vector_store(video_id=signature):
            st.session_state.vector_index_signature = signature
            st.session_state.vector_index_built = True
            return True

    # Build a new index if loading failed or signature doesn't match
    try:
        with st.spinner("📚 正在构建向量索引..."):
            rag_system.build_vector_store(segments, video_id=signature)
        st.session_state.vector_index_signature = signature
        st.session_state.vector_index_built = True
        return True
    except Exception as e:
        st.session_state.vector_index_built = False
        if "vector_index_signature" in st.session_state:
            del st.session_state["vector_index_signature"]
        # 重新抛出异常，让调用者处理
        raise

