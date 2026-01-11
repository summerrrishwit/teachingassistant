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
    - **📌 要点提取**: 提取关键概念和要点
    - **❓ 问答对生成**: 转换为学习问答对
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
            <strong>支持格式:</strong> MP4, WebM, MOV | 
            <strong>最大文件大小:</strong> 200MB
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_upload_success():
    """Upload success banner - 优化版本."""
    st.markdown("""
    <div class="upload-success">
        <h4>✅ 视频上传成功！</h4>
        <p>文件已准备就绪，请选择分析模式</p>
    </div>
    """, unsafe_allow_html=True)


def render_mode_cards():
    """Analysis mode cards, returns tuple of (summary_clicked, qa_clicked, bullet_points_clicked, qa_style_clicked)."""
    st.markdown("---")
    st.markdown("### 🎯 选择分析模式")
    
    # 使用一行四列布局
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="analysis-mode-card">
            <h3>📝 完整视频总结</h3>
            <p>
                对整个视频进行综合分析，包括：<br>
                • 视频概述和主要话题<br>
                • 关键洞察和学习要点<br>
                • 目标受众分析<br>
                • 学习目标总结
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        summary_clicked = st.button("🚀 开始完整分析", key="summary_btn", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="analysis-mode-card">
            <h3>❓ 特定时间戳问答</h3>
            <p>
                在视频的特定时间点提问，获得精准答案：<br>
                • 指定时间戳进行提问<br>
                • 结合视频帧和转录文本<br>
                • 支持多轮对话<br>
                • 生成学习笔记
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        qa_clicked = st.button("🎯 开始时间戳问答", key="qa_btn", use_container_width=True)
    
    with col3:
        st.markdown("""
        <div class="analysis-mode-card">
            <h3>📌 视频要点提取</h3>
            <p>
                从视频中提取关键概念和要点：<br>
                • 自动识别核心概念<br>
                • 生成结构化要点列表<br>
                • 包含详细解释说明<br>
                • 便于快速复习
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        bullet_points_clicked = st.button("📋 开始提取要点", key="bullet_points_btn", use_container_width=True)
    
    with col4:
        st.markdown("""
        <div class="analysis-mode-card">
            <h3>❓ 问答对生成</h3>
            <p>
                将视频内容转换为问答对格式：<br>
                • 自动生成问题与答案<br>
                • 覆盖不同难度层次<br>
                • 适合学习与复习<br>
                • 支持导出保存
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        qa_style_clicked = st.button("📝 开始生成问答对", key="qa_style_btn", use_container_width=True)
    
    return summary_clicked, qa_clicked, bullet_points_clicked, qa_style_clicked

