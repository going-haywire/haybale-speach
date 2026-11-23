"""STT_MODEL: a speech-to-text model loaded on a device, ready to transcribe."""

from typing import Any, ClassVar

from haywire.api import DataWire, PrimitiveField, PrimitiveType, StoreStrategy, type as type_decorator

from haybale_speech.value.stt_model import SttModel


@type_decorator(
    wire_type=DataWire,
    label="STT Model",
    description="A speech-to-text model loaded on a device, ready to run. Never saved with the graph.",
    color="#8a7ad9",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class STT_MODEL(PrimitiveType[SttModel | None]):
    """A speech-to-text model loaded on a device, ready to transcribe.

    Load STT Model gives it; Transcribe takes it.
    """

    #: The catalog fields a model of this kind must match, for the model picker.
    catalog_filter: ClassVar[dict[str, list[Any]] | None] = {"purpose": ["stt"]}

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self._value = _as_model(self._value)  # the type's default arrives as JSON

    @classmethod
    def fits(cls, model: SttModel) -> bool:
        """Return whether *model* turns speech into text."""
        return isinstance(model, SttModel)

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"value": {}}``: a loaded model is never serialized."""
        return {"value": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> None:
        """Return ``None``: a loaded model never round-trips."""
        return None


class STT_MODELField(PrimitiveField):
    """DataField for STT_MODEL: an ``SttModel``, or ``None`` until Load STT Model ran."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._default = _as_model(self._default)
        self._value = self._default

    def _set_own(self, value: Any) -> None:
        self._value = _as_model(value)


def _as_model(value: Any) -> SttModel | None:
    return value if isinstance(value, SttModel) else None


STT_MODEL.field_class = STT_MODELField
