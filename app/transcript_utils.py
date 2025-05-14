import whisper
from typing import List, Dict
from singleton_class import Singleton

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

def get_transcript_around(segments: List[Dict], timestamp: float, window: int = 5) -> str:
    """
    Extract transcript within ±`window` seconds around the given timestamp.
    """
    context = []
    for seg in segments:
        if abs(seg['start'] - timestamp) <= window or abs(seg['end'] - timestamp) <= window:
            context.append(seg['text'])
    return ' '.join(context)
