import streamlit as st
import os
from pathlib import Path
from video_utils import save_uploaded_video, extract_frames_around, extract_key_frames_for_summary
from transcript_utils import WhisperModel, get_transcript_around, get_transcript_full, summarize_transcript
from config import VIDEO_PATH, FRAME_DIR
from llm_utils import get_response
import re

def parse_timestamp(input_str):
    """Convert HH:MM:SS, MM:SS or SS to seconds"""
    input_str = input_str.strip()
    parts = input_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        elif len(parts) == 1:
            return int(parts[0])
        else:
            return None
    except ValueError:
        return None
st.set_page_config(
    page_title="AI Video Assistant", 
    page_icon="🎥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .analysis-mode-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .analysis-mode-card:hover {
        border-color: #667eea;
        transform: scale(1.02);
    }
    .progress-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .result-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        color: #333333;
    }
    .result-container h1, .result-container h2, .result-container h3, .result-container h4, .result-container h5, .result-container h6 {
        color: #333333;
    }
    .result-container p, .result-container li, .result-container div {
        color: #333333;
    }
    .button-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with app information
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🎥 AI Video Assistant</h2>
        <p style="color: #666;">智能视频分析助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 主要功能")
    st.markdown("""
    - **📝 完整视频总结**: 综合分析整个视频内容
    - **❓ 时间戳问答**: 在特定时间点提问
    - **🖼️ 关键帧提取**: 自动提取重要画面
    - **📊 多模态分析**: 结合视频和音频内容
    - **📥 结果导出**: 支持多种格式导出
    """)
    
    st.markdown("### 💡 使用提示")
    st.markdown("""
    1. 上传视频文件（支持MP4、WebM、MOV）
    2. 选择分析模式
    3. 等待AI分析完成
    4. 查看分析结果
    5. 导出或保存结果
    """)
    
    st.markdown("### 🔧 技术特性")
    st.markdown("""
    - **Whisper**: 语音识别转录
    - **Ollama**: 大语言模型分析
    - **OpenCV**: 视频帧提取
    - **Streamlit**: 现代化Web界面
    """)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎥 AI Video Assistant</h1>
    <p>智能视频分析助手 - 让AI帮您理解视频内容</p>
</div>
""", unsafe_allow_html=True)

# Step 1: Upload video with improved UI
st.markdown("### 📁 上传视频文件")
st.markdown("""
<div class="feature-card">
    <p style="text-align: center; color: #666; margin-bottom: 1rem;">
        支持格式: MP4, WebM, MOV | 最大文件大小: 200MB
    </p>
</div>
""", unsafe_allow_html=True)

video_file = st.file_uploader(
    "选择视频文件", 
    type=["mp4", "webm", "mov"],
    help="上传您想要分析的视频文件"
)

if video_file:
    save_uploaded_video(video_file, VIDEO_PATH)
    
    # Clear RAG index when new video is uploaded
    if "vector_index_built" in st.session_state:
        del st.session_state.vector_index_built
    from llm_utils import get_rag_system
    rag_system = get_rag_system()
    rag_system.vectorstore = None
    
    # Success message with better styling
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    ">
        <h4 style="margin: 0;">✅ 视频上传成功！</h4>
        <p style="margin: 0.5rem 0 0 0;">文件已准备就绪，请选择分析模式</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Video player with better styling
    st.markdown("### 🎬 视频预览")
    st.video(str(VIDEO_PATH))

    # Add video analysis options with improved UI
    st.markdown("---")
    st.markdown("### 🎯 选择分析模式")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="analysis-mode-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📝 完整视频总结</h3>
            <p style="color: #666; font-size: 16px; line-height: 1.6;">
                对整个视频进行综合分析，包括：<br>
                • 视频概述和主要话题<br>
                • 关键洞察和学习要点<br>
                • 目标受众分析<br>
                • 学习目标总结
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 开始完整分析", key="summary_btn", type="primary"):
            st.session_state.analysis_mode = "summary"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="analysis-mode-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">❓ 特定时间戳问答</h3>
            <p style="color: #666; font-size: 16px; line-height: 1.6;">
                在视频的特定时间点提问，获得精准答案：<br>
                • 指定时间戳进行提问<br>
                • 结合视频帧和转录文本<br>
                • 支持多轮对话<br>
                • 生成学习笔记
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 开始时间戳问答", key="qa_btn"):
            st.session_state.analysis_mode = "qa"
            st.rerun()
    
    # Initialize analysis mode if not set
    if "analysis_mode" not in st.session_state:
        st.session_state.analysis_mode = None

    # Add back button if in analysis mode
    if st.session_state.analysis_mode is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← 返回选择模式", key="back_to_mode_selection", type="secondary"):
                st.session_state.analysis_mode = None
                st.rerun()

    # Handle different analysis modes
    if st.session_state.analysis_mode == "summary":
        # Complete Video Summary Mode
        st.markdown("---")
        st.markdown("### 📝 完整视频总结分析")
        
        # Check if analysis already exists
        if "video_analysis" in st.session_state and "summary_frames" in st.session_state:
            analysis_time = st.session_state.get('analysis_time', '未知时间')
            st.info(f"💡 检测到已生成的视频分析（生成时间: {analysis_time}），直接显示结果。如需重新生成，请点击下方按钮。")
            
            # Display existing results with better styling
            st.markdown("### 📊 视频分析结果")
            
            # Use expander for better organization
            with st.expander("📋 查看完整分析结果", expanded=True):
                st.markdown(st.session_state.video_analysis)
            
            # Show key frames with better styling
            st.markdown("### 🖼️ 关键视频帧")
            
            # Use expander for key frames
            with st.expander("🖼️ 查看关键视频帧", expanded=True):
                cols = st.columns(len(st.session_state.summary_frames))
                for i, frame_path in enumerate(st.session_state.summary_frames):
                    with cols[i]:
                        st.image(frame_path, caption=f"关键帧 {i+1}")
            
            # Action buttons
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                if st.button("🔄 重新生成", key="regenerate_summary", type="secondary"):
                    # Clear existing analysis
                    if "video_analysis" in st.session_state:
                        del st.session_state.video_analysis
                    if "summary_frames" in st.session_state:
                        del st.session_state.summary_frames
                    if "processing_summary" in st.session_state:
                        st.session_state.processing_summary = False
                    st.rerun()
            
            with col2:
                # Export analysis as text file
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
                    # Clear all analysis data
                    keys_to_clear = ["video_analysis", "summary_frames", "full_transcript", "analysis_time", "processing_summary"]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        else:
            # Generate new analysis
            # Initialize processing state
            if "processing_summary" not in st.session_state:
                st.session_state.processing_summary = False
            
            if st.button("🚀 生成完整视频分析", key="generate_summary", type="primary"):
                st.session_state.processing_summary = True
                st.rerun()
            
            # Show progress if processing
            if st.session_state.processing_summary:
                st.markdown("""
                <div class="progress-container">
                    <h3 style="color: #667eea; margin-bottom: 1rem;">📊 分析进度</h3>
                </div>
                """, unsafe_allow_html=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Transcribe the entire video
                    status_text.text("🔍 正在转录视频...")
                    progress_bar.progress(20)
                    whisper_model = WhisperModel()
                    segments = whisper_model.transcribe(VIDEO_PATH)
                    full_transcript = get_transcript_full(segments)
                    
                    # Step 2: Extract key frames from the entire video
                    status_text.text("🖼️ 正在提取关键帧...")
                    progress_bar.progress(50)
                    summary_frames = extract_key_frames_for_summary(VIDEO_PATH, FRAME_DIR, num_frames=5)
                    
                    # Step 2.5: Build RAG vector index (optional for summary mode)
                    status_text.text("📚 正在构建智能检索索引...")
                    progress_bar.progress(60)
                    from llm_utils import get_rag_system
                    rag_system = get_rag_system()
                    if "vector_index_built" not in st.session_state or not st.session_state.vector_index_built:
                        try:
                            rag_system.build_vector_store(segments)
                            st.session_state.vector_index_built = True
                        except Exception as e:
                            st.warning(f"⚠️ 向量索引构建失败: {e}")
                            st.session_state.vector_index_built = False
                    
                    # Step 3: Generate comprehensive analysis
                    status_text.text("🤖 正在生成综合分析...")
                    progress_bar.progress(80)
                    video_analysis = get_response(
                        question="Analyze the entire video", 
                        text="", 
                        full_transcript=full_transcript, 
                        prompt_key="video_summary", 
                        summarized_transcript=full_transcript,
                        segments=None,  # 总结模式不需要RAG
                        use_rag=False
                    )
                    
                    # Complete
                    status_text.text("✅ 分析完成！")
                    progress_bar.progress(100)
                    
                    # Save to session state
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
    
    elif st.session_state.analysis_mode == "qa":
        # Timestamp-based Q&A mode with RAG
        st.markdown("---")
        st.markdown("### ❓ 特定时间戳问答")
        st.markdown("""
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <p style="margin: 0; color: #1976d2;">
                💡 <strong>使用说明</strong>：输入时间戳和问题，系统将基于时间戳和问题进行 RAG 检索，生成精准答案。
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize conversation history if not exists
        if "qa_conversation_history" not in st.session_state:
            st.session_state.qa_conversation_history = []
        
        # Main form for timestamp and question
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
                    # Parse timestamp
                    seconds = parse_timestamp(timestamp_input)
                    if seconds is None:
                        st.error("❌ 时间戳格式错误。请使用 HH:MM:SS, MM:SS, 或 SS 格式")
                    else:
                        manual_time = seconds
                        timestamp_display = f"{int(manual_time//60)}:{int(manual_time%60):02d}" if manual_time >= 60 else f"{int(manual_time)}秒"
                        
                        # Process the question with RAG
                        with st.spinner("🔍 正在处理..."):
                            try:
                                # Step 1: Transcribe video (if not already done)
                                if "qa_segments" not in st.session_state or not st.session_state.get("qa_segments"):
                                    with st.spinner("📝 正在转录视频..."):
                                        whisper_model = WhisperModel()
                                        segments = whisper_model.transcribe(VIDEO_PATH)
                                        st.session_state.qa_segments = segments
                                        st.session_state.qa_full_transcript = get_transcript_full(segments)
                                        
                                        # Summarize if too long
                                        full_transcript = st.session_state.qa_full_transcript
                                        if len(full_transcript.split()) > 1000:
                                            st.session_state.qa_summarized_transcript = summarize_transcript(full_transcript, word_limit=1000)
                                        else:
                                            st.session_state.qa_summarized_transcript = full_transcript
                                else:
                                    segments = st.session_state.qa_segments
                                    full_transcript = st.session_state.qa_full_transcript
                                
                                # Step 2: Build RAG vector index (if not already built)
                                from llm_utils import get_rag_system
                                rag_system = get_rag_system()
                                if "vector_index_built" not in st.session_state or not st.session_state.vector_index_built:
                                    with st.spinner("📚 正在构建智能检索索引..."):
                                        try:
                                            rag_system.build_vector_store(segments)
                                            st.session_state.vector_index_built = True
                                        except Exception as e:
                                            st.warning(f"⚠️ 向量索引构建失败: {e}")
                                            st.session_state.vector_index_built = False
                                
                                # Step 3: Get transcript snippet around timestamp
                                transcript_snippet = get_transcript_around(segments, manual_time, window=5)
                                
                                # Step 4: Extract frames around timestamp
                                with st.spinner("🖼️ 正在提取视频帧..."):
                                    frames = extract_frames_around(VIDEO_PATH, manual_time, FRAME_DIR)
                                
                                # Step 5: Generate answer using RAG
                                with st.spinner("🤖 正在生成答案（使用 RAG 检索）..."):
                                    answer = get_response(
                                        question=question,
                                        text=transcript_snippet,
                                        full_transcript=st.session_state.qa_full_transcript,
                                        prompt_key="video_qa",
                                        summarized_transcript=st.session_state.qa_summarized_transcript,
                                        segments=segments,  # 传递segments用于RAG检索
                                        use_rag=True  # 启用RAG
                                    )
                                
                                # Step 6: Save to conversation history
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
        
        # Display conversation history
        if st.session_state.qa_conversation_history:
            st.markdown("---")
            st.markdown("### 💬 问答历史")
            
            for idx, item in enumerate(reversed(st.session_state.qa_conversation_history[-10:])):  # 显示最近10条
                with st.expander(f"⏱️ {item['timestamp_display']} - Q{len(st.session_state.qa_conversation_history) - idx}: {item['question'][:50]}...", expanded=(idx == 0)):
                    st.markdown(f"**⏱️ 时间戳:** {item['timestamp_display']} ({item['timestamp']} 秒)")
                    st.markdown(f"**❓ 问题:** {item['question']}")
                    st.markdown("**📄 相关转录片段:**")
                    st.info(item['transcript_snippet'][:200] + "..." if len(item['transcript_snippet']) > 200 else item['transcript_snippet'])
                    st.markdown("**🤖 回答:**")
                    st.markdown(item['answer'])
            
            # Action buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ 清除历史", key="clear_qa_history", type="secondary"):
                    st.session_state.qa_conversation_history = []
                    # Clear RAG index state but keep the index itself (can be reused)
                    if "vector_index_built" in st.session_state:
                        del st.session_state.vector_index_built
                    st.rerun()
            
            with col2:
                # Export conversation history
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