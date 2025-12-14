import streamlit as st


def render_sidebar():
    """Sidebar content with feature list and tips."""
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


def render_header():
    """Main header section."""
    st.markdown("""
    <div class="main-header">
        <h1>🎥 AI Video Assistant</h1>
        <p>智能视频分析助手 - 让AI帮您理解视频内容</p>
    </div>
    """, unsafe_allow_html=True)


def render_upload_card():
    """Upload hint card."""
    st.markdown("### 📁 上传视频文件")
    st.markdown("""
    <div class="feature-card">
        <p style="text-align: center; color: #666; margin-bottom: 1rem;">
            支持格式: MP4, WebM, MOV | 最大文件大小: 200MB
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_upload_success():
    """Upload success banner."""
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


def render_mode_cards():
    """Analysis mode cards, returns tuple of (summary_clicked, qa_clicked)."""
    st.markdown("---")
    st.markdown("### 🎯 选择分析模式")
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
        
        summary_clicked = st.button("🚀 开始完整分析", key="summary_btn", type="primary")
    
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
        
        qa_clicked = st.button("🎯 开始时间戳问答", key="qa_btn")
    
    return summary_clicked, qa_clicked
