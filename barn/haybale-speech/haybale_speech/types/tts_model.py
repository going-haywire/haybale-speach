"""TTS_MODEL: a text-to-speech model loaded on a device, ready to speak."""

from typing import Any, ClassVar

from haywire.api import DataWire, StoreStrategy, type as type_decorator

from haybale_speech.types.speech_model import SPEECH_MODEL, SPEECH_MODELField
from haybale_speech.value.tts_model import TtsModel


@type_decorator(
    wire_type=DataWire,
    label="TTS Model",
    description="A text-to-speech model loaded on a device, ready to run. Never saved with the graph.",
    color="#7a9ad9",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class TTS_MODEL(SPEECH_MODEL):
    """A text-to-speech model loaded on a device, ready to speak.

    Load Speech Model gives it when wired to Speak or Pick Voice; they take it.
    """

    catalog_filter: ClassVar[dict[str, list[Any]] | None] = {"purpose": ["tts"]}
    purpose: ClassVar[str | None] = "tts"

    @classmethod
    def fits(cls, model: Any) -> bool:
        """Return whether *model* turns text into speech."""
        return isinstance(model, TtsModel)


class TTS_MODELField(SPEECH_MODELField):
    """DataField for TTS_MODEL: only a model of that kind fits."""

    kind = TTS_MODEL


TTS_MODEL.field_class = TTS_MODELField
