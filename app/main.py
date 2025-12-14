import streamlit as st
from video_utils import save_uploaded_video
from config import VIDEO_PATH, FRAME_DIR
from workflows import clear_vector_index_state, handle_summary_mode, handle_qa_mode
from ui import render_sidebar, render_header, render_upload_card, render_upload_success, render_mode_cards
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
    summary_clicked, qa_clicked = render_mode_cards()
    if summary_clicked:
        st.session_state.analysis_mode = "summary"
        st.rerun()
    if qa_clicked:
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
        handle_summary_mode(VIDEO_PATH, FRAME_DIR)
    elif st.session_state.analysis_mode == "qa":
        handle_qa_mode(VIDEO_PATH, FRAME_DIR)
