"""Load STT Model — the speech-to-text loader of 0.0.2, kept so saved graphs load. Use Load Speech Model."""

from __future__ import annotations

from typing import Any

from haywire.api import NodeType, node

from haybale_speech.nodes.load_speech_model import NOT_LOADED, BaseSpeechLoaderNode

__all__ = ["NOT_LOADED", "LoadSttModelNode"]


@node(
    label="Load STT Model",
    description="Loads a speech-to-text model onto the CPU or GPU, for Transcribe",
    menu="speech",
    search_tags=["model", "load", "speech", "stt", "asr", "whisper", "faster-whisper", "transcribe"],
    node_type=NodeType.DATA,
    deprecation_warning="Load STT Model is replaced by Load Speech Model, which loads every kind of speech model",
)
class LoadSttModelNode(BaseSpeechLoaderNode):
    """Loads a speech-to-text model onto the CPU or GPU, for Transcribe. Replaced by Load Speech Model.

    It works as before, and graphs saved with it keep loading. New graphs use Load Speech
    Model, which takes the kind of model the node it feeds needs.

    Pick the model on the card: a catalog entry, such as *Whisper small*. The settings
    `Device`, `Compute type`, `CPU threads` and `Workers` are those of Load Speech Model.
    """

    def init(self) -> None:
        from haybale_speech.types.stt_model import STT_MODEL

        self._add_model_inlet(STT_MODEL)
        self.add(STT_MODEL.as_outlet("loaded", label="Model", description="The loaded model"))

    def _outlet_kind(self) -> Any:
        from haybale_speech.types.stt_model import STT_MODEL

        return STT_MODEL
