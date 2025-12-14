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

Important: Do NOT include any promotional content, subscription links, newsletter mentions, website URLs, or advertising in your response. Only provide the answer to the question based on the video content.

Global Video Context:
{global_context}

Focused Transcript Snippet:
{text}

Question: {question}
Answer:"""

PROMPT_QA_RAG = """You are a helpful teaching assistant. You will be provided with:
- A few video frames (images)
- A focused transcript snippet around a specific timestamp
- Retrieved relevant contexts from the video (with timestamps, based on semantic similarity)
- A summary of the video as background

Your job is to answer the question using all provided information. When referencing specific content, mention the timestamp if available.

Important: Do NOT include any promotional content, subscription links, newsletter mentions, website URLs, or advertising in your response. Only provide the answer to the question based on the video content.

Retrieved Relevant Contexts (语义检索到的相关片段):
{retrieved_contexts}

Global Video Summary:
{global_context}

Focused Transcript Snippet (around timestamp):
{text}

Question: {question}
Answer:"""

PROMPT_BULLET = """You are a helpful teaching assistant. You will be provided with a full transcript of the video.
Your job is to generate bullet points for the concepts discussed in the video with proper explanation of each step.

Important: Do NOT include any promotional content, subscription links, newsletter mentions, website URLs, or advertising in your response. Only extract the key concepts and explanations from the video content.

Full Transcript:
{text}
Answer in bullet points:"""

PROMPT_QA_STYLE = """You are a helpful teaching assistant. You will be given transcript of the video.
Your job is to convert the video content into a set of question-answer pairs for study.

Important: Do NOT include any promotional content, subscription links, newsletter mentions, website URLs, or advertising in your response. Only create question-answer pairs based on the video content.

Full Transcript:
{text}
Answer in question-answer pairs:
"""

PROMPT_VIDEO_SUMMARY = """You are a helpful teaching assistant. You will be provided with:
- A complete transcript of the video
- Key video frames extracted from different time points

Your job is to provide a comprehensive analysis and summary of the entire video content.

Important: Do NOT include any promotional content, subscription links, newsletter mentions, website URLs, or advertising in your response. Only analyze and summarize the actual video content.

Video Transcript:
{text}

Please provide:
1. **Video Overview**: A brief summary of what the video is about
2. **Main Topics**: List the key topics and concepts discussed
3. **Key Insights**: Important takeaways and insights from the video
4. **Structure Analysis**: How the content is organized and presented
5. **Target Audience**: Who would benefit from watching this video
6. **Learning Objectives**: What viewers can expect to learn

Please format your response in a clear, structured manner with proper headings and bullet points.
"""

prompt_dict = {
    "video_qa": PROMPT_QA,
    "video_qa_rag": PROMPT_QA_RAG,
    "bullet_points": PROMPT_BULLET,
    "qa_style": PROMPT_QA_STYLE,
    "video_summary": PROMPT_VIDEO_SUMMARY
}
