import streamlit as st
import os
from pathlib import Path
from video_utils import save_uploaded_video, extract_frames_around
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
st.set_page_config(page_title="Video QA App", layout="centered")
st.title("🎥 Ask Questions on Video")

# Step 1: Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "webm", "mov"])
if video_file:
    save_uploaded_video(video_file, VIDEO_PATH)
    st.success("Video uploaded successfully.")
    st.video(str(VIDEO_PATH))

    # Step 2: Ask for timestamp
    timestamp_input = st.text_input("⏱️ Enter timestamp (HH:MM:SS, MM:SS, or SS):")


    manual_time = 0
    if timestamp_input:
        seconds = parse_timestamp(timestamp_input)
        if seconds is None:
            st.error("❌ Invalid timestamp format. Try HH:MM:SS, MM:SS, or just seconds.")
        else:
            manual_time = seconds
        st.success(f"✅ Parsed timestamp: {manual_time} seconds")
        # Step 3: Transcript
        with st.spinner("🔍 Transcribing video..."):
            whisper_model = WhisperModel()
            segments = whisper_model.transcribe(VIDEO_PATH)
            full_transcript = get_transcript_full(segments)
            summarized_transcript = full_transcript
            if len(full_transcript.split()) > 1000:
                summarized_transcript = summarize_transcript(full_transcript, word_limit=1000)
           
            transcript_snippet = get_transcript_around(segments, manual_time, window=5)
        
        st.markdown("### 📄 Transcript Snippet")
        st.write(transcript_snippet)

        # Step 4: Extract frames
        with st.spinner("🖼️ Extracting frames..."):
            frames = extract_frames_around(VIDEO_PATH, manual_time, FRAME_DIR)

        # st.markdown("### 🖼️ Sample Frames")
        # for frame in frames:
        #     st.image(str(frame), use_container_width=True)

        # Initialize session state
        if "conversation" not in st.session_state:
            st.session_state.conversation = []
        if "satisfied" not in st.session_state:
            st.session_state.satisfied = False
        if "current_question" not in st.session_state:
            st.session_state.current_question = ""

        # Ask again only if not satisfied
        if not st.session_state.satisfied:
            with st.form("qa_form"):
                question = st.text_input("❓ Ask your question about this video:", key="question_input")
                submit = st.form_submit_button("Ask")

                if submit and question.strip():
                    with st.spinner("🤖 Model is processing..."):
                        answer = get_response(question=question, text=transcript_snippet, 
                                              full_transcript=full_transcript,
                                              prompt_key="video_qa", summarized_transcript=summarized_transcript)
                        st.session_state.conversation.append((question, answer))
                        st.session_state.current_question = ""  # reset input

        # Show conversation history
        if st.session_state.conversation:
            st.markdown("### 🧠 Conversation")
            for i, (q, a) in enumerate(st.session_state.conversation):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")

        # Satisfied button
        if not st.session_state.satisfied:
            if st.button("✅ I'm satisfied"):
                st.session_state.satisfied = True
                st.success("Conversation ended. Thanks for using the app!")

        # Offer note generation
        if st.session_state.satisfied and "notes_generated" not in st.session_state:
            st.markdown("### 📝 Would you like to generate notes or key concepts from this video?")
            note_style = st.selectbox(
                "Choose the format:",
                ["Bullet Points", "Summary", "Q&A Style"]
            )

            if st.button("🧠 Generate Notes"):
                with st.spinner("✍️ Generating notes..."):
                    if note_style == "Bullet Points":
                        prompt_key = "bullet_points"
                        instruction = "Extract the key concepts as clear bullet points."
                    elif note_style == "Summary":
                        prompt_key = "summary"
                        instruction = "Summarize the video transcript concisely in a paragraph."
                    elif note_style == "Q&A Style":
                        prompt_key = "qa_style"
                        instruction = "Convert the video content into a set of question-answer pairs for study."

                    if prompt_key == "summary":
                        st.session_state.notes_generated = summarized_transcript
                    else:
                        notes = get_response(question="Generate study notes", text="", 
                                            full_transcript=full_transcript, prompt_key=prompt_key, 
                                            summarized_transcript=summarized_transcript)
                    st.session_state.notes_generated = notes
                    st.success("📝 Notes generated!")

        # Show generated notes
        if "notes_generated" in st.session_state:
            st.markdown("### 📌 Generated Notes")
            st.write(st.session_state.notes_generated)