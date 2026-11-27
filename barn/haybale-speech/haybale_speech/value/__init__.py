"""Values carried on haybale-speech's wires."""

from haybale_speech.value.stt_model import SttModel
from haybale_speech.value.transcript import Segment, Transcript
from haybale_speech.value.tts_model import TtsModel
from haybale_speech.value.voice import Voice

#: Every loaded speech model, whatever its kind.
SpeechModel = SttModel | TtsModel

__all__ = ["Segment", "SpeechModel", "SttModel", "Transcript", "TtsModel", "Voice"]
