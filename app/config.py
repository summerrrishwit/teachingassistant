import os
from enum import Enum

VIDEO_PATH = "runtime/uploaded_video.mp4"
FRAME_DIR = "runtime/frames/"
MODEL = "gemma3:4b"
os.makedirs(FRAME_DIR, exist_ok=True)

PROMPT_QA = """You are a helpful teaching assistant. You will be provided with:
- A few video frames (images)
- A focused transcript snippet around a specific timestamp
- Optionally, a summary or full transcript of the video as background

Your job is to answer the question using both the images and transcript context.

Global Video Context:
{global_context}

Focused Transcript Snippet:
{text}

Question: {question}
Answer:"""

PROMPT_BULLET = """You are a helpful teaching assistant. You will be provided with a full transcript of the video.
Your job is to generate bullet points for the concepts discussed in the video with proper explanation of each step.

Full Transcript:
{text}
Answer in bullet points:"""

PROMPT_QA_STYLE = """You are a helpful teaching assistant. You will be given transcript of the video.
Your job is to convert the video content into a set of question-answer pairs for study.

Full Transcript:
{text}
Answer in question-answer pairs:
"""

prompt_dict = {
    "video_qa": PROMPT_QA,
    "bullet_points": PROMPT_BULLET,
    "qa_style": PROMPT_QA_STYLE
}
