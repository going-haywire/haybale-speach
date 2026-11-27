"""SPEECH_MODEL: a speech model of any kind, loaded on a device. STT_MODEL and TTS_MODEL are its kinds."""

from typing import Any, ClassVar

from haywire.api import DataWire, PrimitiveField, PrimitiveType, StoreStrategy, type as type_decorator

from haybale_speech.value import SttModel, TtsModel


@type_decorator(
    wire_type=DataWire,
    label="Speech Model",
    description="A speech model loaded on a device, ready to run. Never saved with the graph.",
    color="#8a7ad9",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class SPEECH_MODEL(PrimitiveType[SttModel | TtsModel | None]):
    """A speech model loaded on a device, of any kind.

    Transcribe takes one kind (`STT_MODEL`), Speak another (`TTS_MODEL`), so a Load Speech Model
    wired to them becomes that kind and offers only the models that fit.
    """

    #: The catalog fields a model of this kind must match, for the model picker.
    catalog_filter: ClassVar[dict[str, list[Any]] | None] = {"purpose": ["stt", "tts"]}

    #: The catalog ``purpose`` of this kind; ``None`` for the base, which takes any.
    purpose: ClassVar[str | None] = None

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self._value = _as_model(self._value)  # the type's default arrives as JSON

    @classmethod
    def fits(cls, model: Any) -> bool:
        """Return whether *model* is a loaded speech model."""
        return isinstance(model, (SttModel, TtsModel))

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"value": {}}``: a loaded model is never serialized."""
        return {"value": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> None:
        """Return ``None``: a loaded model never round-trips."""
        return None


class SPEECH_MODELField(PrimitiveField):
    """DataField for the speech model types: a loaded model, or ``None`` until Load Speech Model ran."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._default = _as_model(self._default)
        self._value = self._default

    def _set_own(self, value: Any) -> None:
        self._value = _as_model(value)


def _as_model(value: Any) -> SttModel | TtsModel | None:
    return value if isinstance(value, (SttModel, TtsModel)) else None


SPEECH_MODEL.field_class = SPEECH_MODELField
