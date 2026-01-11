import whisper
from typing import List, Dict
from app.core import Singleton
from summa.summarizer import summarize
import streamlit as st

class WhisperModel(Singleton):
    model = None

    def __init__(self):
        pass
    
    @classmethod
    def load_model(cls):
        if cls.model is None:
            try:
                cls.model = _get_cached_model()
            except Exception as load_error:
                raise RuntimeError(
                    "Whisper 模型加载失败，请确认已安装依赖并有可用的运行环境。"
                ) from load_error

    @classmethod
    def transcribe(cls, video_path: str) -> List[Dict]:
        """
        Transcribe video and return list of segments with start, end, and text.
        """
        if cls.model is None:
            cls.load_model()
        result = cls.model.transcribe(video_path)
        return result['segments']

@st.cache_resource
def _get_cached_model():
    """Cache Whisper model to avoid repeated heavy loads."""
    return whisper.load_model("base")

def get_transcript_around(segments: List[Dict], timestamp: float, window: int = 60) -> str:
    """
    Extract transcript within ±`window` seconds around the given timestamp.
    """
    context = []
    for seg in segments:
        if abs(seg['start'] - timestamp) <= window or abs(seg['end'] - timestamp) <= window:
            context.append(seg['text'])
    return ' '.join(context)

def get_transcript_full(segments: List[Dict]) -> str:
    """
    Extract full transcript from segments.
    """
    return ' '.join([seg['text'] for seg in segments])

def summarize_transcript(text, word_limit=1000):
    """
    Summarize transcript using TextRank (summa).
    :param text: full transcript
    :param word_limit: maximum words in summary
    :return: summary string
    """
    return summarize(text, words=word_limit, split=False)

