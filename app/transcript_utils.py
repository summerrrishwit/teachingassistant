import whisper
from typing import List, Dict
from singleton_class import Singleton
from summa.summarizer import summarize

class WhisperModel(Singleton):
    model = None

    def __init__(self):
        pass
    
    @classmethod
    def load_model(cls):
        if cls.model is None:
            cls.model = whisper.load_model("base")

    @classmethod
    def transcribe(cls, video_path: str) -> List[Dict]:
        """
        Transcribe video and return list of segments with start, end, and text.
        """
        result = cls.model.transcribe(video_path)
        return result['segments']

WhisperModel.load_model()

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
