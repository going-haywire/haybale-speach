"""STT_MODEL: a speech-to-text model loaded on a device, ready to transcribe."""

from typing import Any, ClassVar

from haywire.api import DataWire, StoreStrategy, type as type_decorator

from haybale_speech.types.speech_model import SPEECH_MODEL, SPEECH_MODELField
from haybale_speech.value.stt_model import SttModel


@type_decorator(
    wire_type=DataWire,
    label="STT Model",
    description="A speech-to-text model loaded on a device, ready to run. Never saved with the graph.",
    color="#8a7ad9",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class STT_MODEL(SPEECH_MODEL):
    """A speech-to-text model loaded on a device, ready to transcribe.

    Load Speech Model gives it when wired to Transcribe; Transcribe takes it.
    """

    catalog_filter: ClassVar[dict[str, list[Any]] | None] = {"purpose": ["stt"]}
    purpose: ClassVar[str | None] = "stt"

    @classmethod
    def fits(cls, model: Any) -> bool:
        """Return whether *model* turns speech into text."""
        return isinstance(model, SttModel)


class STT_MODELField(SPEECH_MODELField):
    """DataField for STT_MODEL: only a model of that kind fits."""

    kind = STT_MODEL


STT_MODEL.field_class = STT_MODELField
