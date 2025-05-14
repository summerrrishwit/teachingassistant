from pathlib import Path
import os

VIDEO_PATH = "runtime/uploaded_video.mp4"
FRAME_DIR = "runtime/frames/"
os.makedirs(FRAME_DIR, exist_ok=True)
PROMPT = """You are a teaching assistant. You will be provided images of frames from a video along with the transcript 
that will contain information extracted from a particular timestamp. Your task is to answer the question correctly 
using the images and the transcript. If you do not know the answer respond with 'I do not know'.

Transcript:
{text}

Question: {question}
"""