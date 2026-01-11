import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，确保可以导入 app 模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from app.video_utils import save_uploaded_video
from app.config import VIDEO_PATH, FRAME_DIR
from app.services.vector_service import clear_vector_index_state
from app.workflows import (
    handle_summary_mode,
    handle_qa_mode,
    handle_bullet_points_mode,
    handle_qa_style_mode
)
from app.ui import (
    render_sidebar,
    render_header,
    render_upload_card,
    render_upload_success,
    render_mode_cards
)
from app.styles.css import MAIN_CSS

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(MAIN_CSS, unsafe_allow_html=True)

# Sidebar with app information
with st.sidebar:
    render_sidebar()

# Main header
render_header()

# Step 1: Upload video with improved UI
render_upload_card()

video_file = st.file_uploader(
    "选择视频文件", 
    type=["mp4", "webm", "mov"],
    help="上传您想要分析的视频文件"
)

if video_file:
    save_uploaded_video(video_file, VIDEO_PATH)
    
    # Clear RAG index when new video is uploaded
    clear_vector_index_state()
    
    # Success message with better styling
    render_upload_success()
    
    # Video player with better styling
    st.markdown("### 🎬 视频预览")
    st.video(str(VIDEO_PATH))

    # Add video analysis options with improved UI
    summary_clicked, qa_clicked, bullet_points_clicked, qa_style_clicked = render_mode_cards()
    if summary_clicked:
        st.session_state.analysis_mode = "summary"
        st.rerun()
    if qa_clicked:
        st.session_state.analysis_mode = "qa"
        st.rerun()
    if bullet_points_clicked:
        st.session_state.analysis_mode = "bullet_points"
        st.rerun()
    if qa_style_clicked:
        st.session_state.analysis_mode = "qa_style"
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
        handle_summary_mode(VIDEO_PATH, FRAME_DIR)
    elif st.session_state.analysis_mode == "qa":
        handle_qa_mode(VIDEO_PATH, FRAME_DIR)
    elif st.session_state.analysis_mode == "bullet_points":
        handle_bullet_points_mode(VIDEO_PATH, FRAME_DIR)
    elif st.session_state.analysis_mode == "qa_style":
        handle_qa_style_mode(VIDEO_PATH, FRAME_DIR)

