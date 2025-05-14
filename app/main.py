import streamlit as st
import os
from pathlib import Path
from video_utils import save_uploaded_video, extract_frames_around
from transcript_utils import WhisperModel, get_transcript_around
from config import VIDEO_PATH, FRAME_DIR
from llm_utils import get_response

st.set_page_config(page_title="Video QA App", layout="centered")
st.title("🎥 Ask Questions on Video")

# Step 1: Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "webm", "mov"])
if video_file:
    save_uploaded_video(video_file, VIDEO_PATH)
    st.success("Video uploaded successfully.")
    st.video(str(VIDEO_PATH))

    # Step 2: Ask for timestamp
    manual_time = st.number_input("⏸️ Enter the timestamp you paused at (in seconds):", step=1.0)

    if manual_time:
        # Step 3: Transcript
        with st.spinner("🔍 Transcribing video..."):
            whisper_model = WhisperModel()
            segments = whisper_model.transcribe(VIDEO_PATH)
            transcript_snippet = get_transcript_around(segments, manual_time, window=5)
        
        st.markdown("### 📄 Transcript Snippet")
        st.write(transcript_snippet)

        # Step 4: Extract frames
        with st.spinner("🖼️ Extracting frames..."):
            frames = extract_frames_around(VIDEO_PATH, manual_time, FRAME_DIR)

        st.markdown("### 🖼️ Sample Frames")
        for frame in frames:
            st.image(str(frame), use_column_width=True)

        # Optional: Ask question
        question = st.text_input("❓ Ask your question based on the video:")
        if question:
            st.markdown("### 🤖 Answer")

            # Here you would typically call a model to get the answer
            # For now, we will just echo the question
            st.write(f"You asked: {question}")
            st.write("Model is processing your question...")
            # Simulate model response
            # In a real application, you would call the model here
            answer = get_response(question, transcript_snippet)
            st.write(answer)
            st.write("Model response: [Simulated response]")