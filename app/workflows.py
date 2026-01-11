from pathlib import Path
import re
import streamlit as st
from app.llm_utils import get_response
from app.video_utils import extract_key_frames_for_summary, extract_frames_around
from app.transcript_utils import (
    WhisperModel,
    get_transcript_full,
    summarize_transcript,
    get_transcript_around
)
from app.services.vector_service import ensure_vector_index
from app.utils.timestamp import parse_timestamp


def markdown_to_html_simple(text: str) -> str:
    """Convert basic markdown to HTML (supports bold, italic, code, links)."""
    # Escape HTML special characters first
    html = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Convert markdown to HTML
    # Bold: **text** or __text__
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
    
    # Italic: *text* or _text_
    html = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', html)
    html = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<em>\1</em>', html)
    
    # Code: `code`
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
    
    # Links: [text](url)
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2" target="_blank">\1</a>', html)
    
    # Line breaks: \n -> <br>
    html = html.replace('\n', '<br>')
    
    return html


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
    """Render and execute timestamp QA workflow with GPT-style chat interface."""
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

    # 显示对话历史（GPT风格）
    if st.session_state.qa_conversation_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for idx, item in enumerate(st.session_state.qa_conversation_history):
            # 用户消息（右侧）
            question_escaped = item['question'].replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            user_message_html = f"""
            <div class="message-wrapper message-user">
                <div class="message-bubble message-bubble-user">
                    <div class="message-meta message-timestamp message-timestamp-user">
                        ⏱️ {item['timestamp_display']}
                    </div>
                    <div class="message-content">
                        {question_escaped}
                    </div>
                </div>
            </div>
            """
            st.markdown(user_message_html, unsafe_allow_html=True)
            
            # AI消息（左侧）
            # 将答案内容转换为HTML，支持markdown格式
            answer_html = markdown_to_html_simple(item['answer'])
            transcript_preview = item['transcript_snippet'][:200] + "..." if len(item['transcript_snippet']) > 200 else item['transcript_snippet']
            transcript_escaped = transcript_preview.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
            
            assistant_message_html = f"""
            <div class="message-wrapper message-assistant">
                <div class="message-bubble message-bubble-assistant">
                    <div class="message-content">
                        {answer_html}
                    </div>
                    <details style="margin-top: 0.75rem; cursor: pointer;">
                        <summary style="color: #666666; font-size: 0.85rem; font-weight: 500;">
                            📄 查看相关转录片段
                        </summary>
                        <div class="transcript-snippet" style="margin-top: 0.5rem;">
                            {transcript_escaped}
                        </div>
                    </details>
                </div>
                <div class="message-meta">
                    🤖 AI助手
                </div>
            </div>
            """
            st.markdown(assistant_message_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 操作按钮
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🗑️ 清除对话", key="clear_qa_history", type="secondary", use_container_width=True):
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
                    label="📥 导出对话",
                    data=export_text,
                    file_name=f"qa_history_{len(st.session_state.qa_conversation_history)}_items.md",
                    mime="text/markdown",
                    type="secondary",
                    use_container_width=True
                )
    else:
        # 空状态提示
        st.markdown("""
        <div class="empty-chat">
            <div class="empty-chat-icon">💬</div>
            <p>还没有对话记录，请在下方输入时间戳和问题开始问答</p>
        </div>
        """, unsafe_allow_html=True)

    # 输入表单（固定在底部）
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    with st.form("timestamp_qa_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            timestamp_input = st.text_input(
                "⏱️ 时间戳",
                placeholder="例如: 1:30",
                key="qa_timestamp_input",
                help="输入视频中的时间点 (HH:MM:SS, MM:SS, 或 SS)"
            )
        with col2:
            question = st.text_input(
                "❓ 您的问题",
                placeholder="例如: 这个时间点讲了什么？",
                key="qa_question_input",
                help="输入您想了解的问题"
            )
        submit = st.form_submit_button("🚀 发送", type="primary", use_container_width=True)

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

                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ 处理过程中出现错误: {str(e)}")
                            st.info("💡 请检查视频文件和时间戳是否正确")
    
    st.markdown('</div>', unsafe_allow_html=True)


def handle_bullet_points_mode(video_path: Path, frame_dir: Path):
    """Render and execute bullet points extraction workflow."""
    st.markdown("---")
    st.markdown("### 📌 视频要点提取")
    st.markdown("""
    <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #1976d2;">
            💡 <strong>使用说明</strong>：从视频转录文本中提取关键概念和要点，生成结构化的学习笔记。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # If already generated, reuse
    if "bullet_points_result" in st.session_state:
        generation_time = st.session_state.get('bullet_points_time', '未知时间')
        st.info(f"💡 检测到已生成的要点提取（生成时间: {generation_time}），直接显示结果。如需重新生成，请点击下方按钮。")

        st.markdown("### 📋 视频要点")
        with st.expander("📌 查看完整要点", expanded=True):
            st.markdown(st.session_state.bullet_points_result)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("🔄 重新生成", key="regenerate_bullet_points", type="secondary"):
                for key in ["bullet_points_result", "bullet_points_transcript", "bullet_points_time", "processing_bullet_points"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            bullet_text = f"""# 视频要点提取

## 生成时间
{st.session_state.get('bullet_points_time', '未知')}

## 视频要点
{st.session_state.bullet_points_result}

## 完整转录文本
{st.session_state.get('bullet_points_transcript', '无')}
"""
            st.download_button(
                label="📥 导出要点",
                data=bullet_text,
                file_name=f"bullet_points_{st.session_state.get('bullet_points_time', 'unknown').replace(':', '-')}.md",
                mime="text/markdown",
                type="secondary"
            )
        with col3:
            if st.button("📋 复制到剪贴板", key="copy_bullet_points", type="secondary"):
                st.code(st.session_state.bullet_points_result, language="markdown")
                st.info("💡 请手动复制上述内容到剪贴板")
        with col4:
            if st.button("🗑️ 清除结果", key="clear_bullet_points", type="secondary"):
                for key in ["bullet_points_result", "bullet_points_transcript", "bullet_points_time", "processing_bullet_points"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        return

    # Generate new bullet points
    if "processing_bullet_points" not in st.session_state:
        st.session_state.processing_bullet_points = False

    if st.button("🚀 生成视频要点", key="generate_bullet_points", type="primary"):
        st.session_state.processing_bullet_points = True
        st.rerun()

    if st.session_state.processing_bullet_points:
        st.markdown("""
        <div class="progress-container">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📊 处理进度</h3>
        </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔍 正在转录视频...")
            progress_bar.progress(30)
            whisper_model = WhisperModel()
            segments = whisper_model.transcribe(video_path)
            full_transcript = get_transcript_full(segments)

            status_text.text("🤖 正在提取要点...")
            progress_bar.progress(70)
            bullet_points = get_response(
                question="Extract key concepts and bullet points",
                text="",
                full_transcript=full_transcript,
                prompt_key="bullet_points",
                summarized_transcript=full_transcript,
                segments=None,
                use_rag=False,
                frame_paths=None
            )

            status_text.text("✅ 要点提取完成！")
            progress_bar.progress(100)

            from datetime import datetime
            st.session_state.bullet_points_result = bullet_points
            st.session_state.bullet_points_transcript = full_transcript
            st.session_state.bullet_points_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.processing_bullet_points = False

            st.success("🎉 视频要点提取完成！")
            st.rerun()

        except Exception as e:
            st.session_state.processing_bullet_points = False
            st.error(f"❌ 处理过程中出现错误: {str(e)}")
            st.info("💡 请尝试重新生成或检查视频文件。")


def handle_qa_style_mode(video_path: Path, frame_dir: Path):
    """Render and execute Q&A style conversion workflow."""
    st.markdown("---")
    st.markdown("### ❓ 问答对生成")
    st.markdown("""
    <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #1976d2;">
            💡 <strong>使用说明</strong>：将视频内容转换为问答对格式，便于学习和复习。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # If already generated, reuse
    if "qa_style_result" in st.session_state:
        generation_time = st.session_state.get('qa_style_time', '未知时间')
        st.info(f"💡 检测到已生成的问答对（生成时间: {generation_time}），直接显示结果。如需重新生成，请点击下方按钮。")

        st.markdown("### 📝 问答对")
        with st.expander("❓ 查看完整问答对", expanded=True):
            st.markdown(st.session_state.qa_style_result)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if st.button("🔄 重新生成", key="regenerate_qa_style", type="secondary"):
                for key in ["qa_style_result", "qa_style_transcript", "qa_style_time", "processing_qa_style"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        with col2:
            qa_text = f"""# 视频问答对

## 生成时间
{st.session_state.get('qa_style_time', '未知')}

## 问答对
{st.session_state.qa_style_result}

## 完整转录文本
{st.session_state.get('qa_style_transcript', '无')}
"""
            st.download_button(
                label="📥 导出问答对",
                data=qa_text,
                file_name=f"qa_pairs_{st.session_state.get('qa_style_time', 'unknown').replace(':', '-')}.md",
                mime="text/markdown",
                type="secondary"
            )
        with col3:
            if st.button("📋 复制到剪贴板", key="copy_qa_style", type="secondary"):
                st.code(st.session_state.qa_style_result, language="markdown")
                st.info("💡 请手动复制上述内容到剪贴板")
        with col4:
            if st.button("🗑️ 清除结果", key="clear_qa_style", type="secondary"):
                for key in ["qa_style_result", "qa_style_transcript", "qa_style_time", "processing_qa_style"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        return

    # Generate new Q&A pairs
    if "processing_qa_style" not in st.session_state:
        st.session_state.processing_qa_style = False

    if st.button("🚀 生成问答对", key="generate_qa_style", type="primary"):
        st.session_state.processing_qa_style = True
        st.rerun()

    if st.session_state.processing_qa_style:
        st.markdown("""
        <div class="progress-container">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📊 处理进度</h3>
        </div>
        """, unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("🔍 正在转录视频...")
            progress_bar.progress(30)
            whisper_model = WhisperModel()
            segments = whisper_model.transcribe(video_path)
            full_transcript = get_transcript_full(segments)

            status_text.text("🤖 正在生成问答对...")
            progress_bar.progress(70)
            qa_pairs = get_response(
                question="Convert video content to Q&A pairs",
                text="",
                full_transcript=full_transcript,
                prompt_key="qa_style",
                summarized_transcript=full_transcript,
                segments=None,
                use_rag=False,
                frame_paths=None
            )

            status_text.text("✅ 问答对生成完成！")
            progress_bar.progress(100)

            from datetime import datetime
            st.session_state.qa_style_result = qa_pairs
            st.session_state.qa_style_transcript = full_transcript
            st.session_state.qa_style_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.processing_qa_style = False

            st.success("🎉 问答对生成完成！")
            st.rerun()

        except Exception as e:
            st.session_state.processing_qa_style = False
            st.error(f"❌ 处理过程中出现错误: {str(e)}")
            st.info("💡 请尝试重新生成或检查视频文件。")
