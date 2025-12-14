import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from llm_utils import get_rag_system, get_response
from video_utils import extract_key_frames_for_summary, extract_frames_around
from transcript_utils import WhisperModel, get_transcript_full, summarize_transcript, get_transcript_around


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


def parse_timestamp(input_str: str) -> Optional[int]:
    """Convert HH:MM:SS, MM:SS or SS to seconds."""
    input_str = input_str.strip()
    parts = input_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        if len(parts) == 1:
            return int(parts[0])
        return None
    except ValueError:
        return None


def handle_summary_mode(video_path: Path, frame_dir: Path):
    """Render and execute summary workflow."""
    st.markdown("---")
    st.markdown("### 📝 完整视频总结分析")

    # If already generated, reuse
    if "video_analysis" in st.session_state and "summary_frames" in st.session_state:
        analysis_time = st.session_state.get('analysis_time', '未知时间')
        st.info(f"💡 检测到已生成的视频分析（生成时间: {analysis_time}），直接显示结果。如需重新生成，请点击下方按钮。")

        st.markdown("### 📊 视频分析结果")
        with st.expander("📋 查看完整分析结果", expanded=True):
            st.markdown(st.session_state.video_analysis)

        st.markdown("### 🖼️ 关键视频帧")
        with st.expander("🖼️ 查看关键视频帧", expanded=True):
            cols = st.columns(len(st.session_state.summary_frames))
            for i, frame_path in enumerate(st.session_state.summary_frames):
                with cols[i]:
                    st.image(frame_path, caption=f"关键帧 {i+1}")

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("🔄 重新生成", key="regenerate_summary", type="secondary"):
                for key in ["video_analysis", "summary_frames", "full_transcript", "analysis_time", "processing_summary"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            analysis_text = f"""# 视频分析报告

## 分析时间
{st.session_state.get('analysis_time', '未知')}

## 视频分析结果
{st.session_state.video_analysis}

## 转录文本
{st.session_state.get('full_transcript', '无')}
"""
            st.download_button(
                label="📥 导出分析",
                data=analysis_text,
                file_name=f"video_analysis_{st.session_state.get('analysis_time', 'unknown')}.md",
                mime="text/markdown",
                type="secondary"
            )
        with col3:
            if st.button("📋 复制到剪贴板", key="copy_summary", type="secondary"):
                st.code(st.session_state.video_analysis, language="markdown")
                st.info("💡 请手动复制上述内容到剪贴板")
        with col4:
            if st.button("🗑️ 清除分析", key="clear_summary", type="secondary"):
                for key in ["video_analysis", "summary_frames", "full_transcript", "analysis_time", "processing_summary"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        return

    # Generate new analysis
    if "processing_summary" not in st.session_state:
        st.session_state.processing_summary = False

    if st.button("🚀 生成完整视频分析", key="generate_summary", type="primary"):
        st.session_state.processing_summary = True
        st.rerun()

    if st.session_state.processing_summary:
        st.markdown("""
        <div class="progress-container">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📊 分析进度</h3>
        </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔍 正在转录视频...")
            progress_bar.progress(20)
            whisper_model = WhisperModel()
            segments = whisper_model.transcribe(video_path)
            full_transcript = get_transcript_full(segments)

            status_text.text("🖼️ 正在提取关键帧...")
            progress_bar.progress(50)
            summary_frames = extract_key_frames_for_summary(video_path, frame_dir, num_frames=5)

            status_text.text("📚 正在构建智能检索索引...")
            progress_bar.progress(60)
            try:
                ensure_vector_index(segments, video_path)
            except Exception as e:
                st.warning(f"⚠️ 向量索引构建失败: {e}")

            status_text.text("🤖 正在生成综合分析...")
            progress_bar.progress(80)
            video_analysis = get_response(
                question="Analyze the entire video",
                text="",
                full_transcript=full_transcript,
                prompt_key="video_summary",
                summarized_transcript=full_transcript,
                segments=None,
                use_rag=False,
                frame_paths=summary_frames
            )

            status_text.text("✅ 分析完成！")
            progress_bar.progress(100)

            from datetime import datetime
            st.session_state.video_analysis = video_analysis
            st.session_state.summary_frames = summary_frames
            st.session_state.full_transcript = full_transcript
            st.session_state.analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.processing_summary = False

            st.success("🎉 完整视频分析生成完成！")
            st.rerun()

        except Exception as e:
            st.session_state.processing_summary = False
            st.error(f"❌ 分析过程中出现错误: {str(e)}")
            st.info("💡 请尝试重新生成或检查视频文件。")


def handle_qa_mode(video_path: Path, frame_dir: Path):
    """Render and execute timestamp QA workflow."""
    st.markdown("---")
    st.markdown("### ❓ 特定时间戳问答")
    st.markdown("""
    <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #1976d2;">
            💡 <strong>使用说明</strong>：输入时间戳和问题，系统将基于时间戳和问题进行 RAG 检索，生成精准答案。
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "qa_conversation_history" not in st.session_state:
        st.session_state.qa_conversation_history = []

    with st.form("timestamp_qa_form", clear_on_submit=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            timestamp_input = st.text_input(
                "⏱️ 时间戳 (HH:MM:SS, MM:SS, 或 SS):",
                placeholder="例如: 1:30 或 90",
                key="qa_timestamp_input",
                help="输入视频中的时间点"
            )
        with col2:
            question = st.text_input(
                "❓ 您的问题:",
                placeholder="例如: 这个时间点讲了什么？",
                key="qa_question_input",
                help="输入您想了解的问题"
            )
        submit = st.form_submit_button("🚀 开始问答", type="primary", use_container_width=True)

        if submit:
            if not timestamp_input or not question.strip():
                st.error("❌ 请同时输入时间戳和问题")
            else:
                seconds = parse_timestamp(timestamp_input)
                if seconds is None:
                    st.error("❌ 时间戳格式错误。请使用 HH:MM:SS, MM:SS, 或 SS 格式")
                else:
                    manual_time = seconds
                    timestamp_display = f"{int(manual_time//60)}:{int(manual_time%60):02d}" if manual_time >= 60 else f"{int(manual_time)}秒"
                    with st.spinner("🔍 正在处理..."):
                        try:
                            if "qa_segments" not in st.session_state or not st.session_state.get("qa_segments"):
                                with st.spinner("📝 正在转录视频..."):
                                    whisper_model = WhisperModel()
                                    segments = whisper_model.transcribe(video_path)
                                    st.session_state.qa_segments = segments
                                    st.session_state.qa_full_transcript = get_transcript_full(segments)
                                    full_transcript = st.session_state.qa_full_transcript
                                    if len(full_transcript.split()) > 1000:
                                        st.session_state.qa_summarized_transcript = summarize_transcript(full_transcript, word_limit=1000)
                                    else:
                                        st.session_state.qa_summarized_transcript = full_transcript
                            else:
                                segments = st.session_state.qa_segments
                                full_transcript = st.session_state.qa_full_transcript

                            with st.spinner("📚 正在构建智能检索索引..."):
                                try:
                                    ensure_vector_index(segments, video_path)
                                except Exception as e:
                                    st.warning(f"⚠️ 向量索引构建失败: {e}")

                            transcript_snippet = get_transcript_around(segments, manual_time, window=5)
                            with st.spinner("🖼️ 正在提取视频帧..."):
                                frames = extract_frames_around(video_path, manual_time, frame_dir)

                            with st.spinner("🤖 正在生成答案（使用 RAG 检索）..."):
                                answer = get_response(
                                    question=question,
                                    text=transcript_snippet,
                                    full_transcript=st.session_state.qa_full_transcript,
                                    prompt_key="video_qa",
                                    summarized_transcript=st.session_state.qa_summarized_transcript,
                                    segments=segments,
                                    use_rag=True,
                                    frame_paths=frames
                                )

                            conversation_item = {
                                'timestamp': manual_time,
                                'timestamp_display': timestamp_display,
                                'question': question,
                                'answer': answer,
                                'transcript_snippet': transcript_snippet
                            }
                            st.session_state.qa_conversation_history.append(conversation_item)

                            st.success(f"✅ 已回答（时间戳: {timestamp_display}）")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ 处理过程中出现错误: {str(e)}")
                            st.info("💡 请检查视频文件和时间戳是否正确")

    if st.session_state.qa_conversation_history:
        st.markdown("---")
        st.markdown("### 💬 问答历史")

        for idx, item in enumerate(reversed(st.session_state.qa_conversation_history[-10:])):
            with st.expander(f"⏱️ {item['timestamp_display']} - Q{len(st.session_state.qa_conversation_history) - idx}: {item['question'][:50]}...", expanded=(idx == 0)):
                st.markdown(f"**⏱️ 时间戳:** {item['timestamp_display']} ({item['timestamp']} 秒)")
                st.markdown(f"**❓ 问题:** {item['question']}")
                st.markdown("**📄 相关转录片段:**")
                st.info(item['transcript_snippet'][:200] + "..." if len(item['transcript_snippet']) > 200 else item['transcript_snippet'])
                st.markdown("**🤖 回答:**")
                st.markdown(item['answer'])

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗑️ 清除历史", key="clear_qa_history", type="secondary"):
                st.session_state.qa_conversation_history = []
                if "vector_index_built" in st.session_state:
                    del st.session_state.vector_index_built
                st.rerun()
        with col2:
            if st.session_state.qa_conversation_history:
                export_text = "# 时间戳问答报告\n\n"
                for idx, item in enumerate(st.session_state.qa_conversation_history):
                    export_text += f"## 问答 {idx + 1}\n\n"
                    export_text += f"**时间戳:** {item['timestamp_display']} ({item['timestamp']} 秒)\n\n"
                    export_text += f"**问题:** {item['question']}\n\n"
                    export_text += f"**相关转录片段:**\n{item['transcript_snippet']}\n\n"
                    export_text += f"**回答:**\n{item['answer']}\n\n---\n\n"

                st.download_button(
                    label="📥 导出问答历史",
                    data=export_text,
                    file_name=f"qa_history_{len(st.session_state.qa_conversation_history)}_items.md",
                    mime="text/markdown",
                    type="secondary"
                )
