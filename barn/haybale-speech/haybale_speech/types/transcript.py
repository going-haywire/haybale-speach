"""TRANSCRIPT: the text of a clip, with its language and timed segments."""

from typing import Any

from haywire.api import DataWire, PrimitiveField, PrimitiveType, StoreStrategy, type as type_decorator

from haybale_speech.value.transcript import Transcript


@type_decorator(
    wire_type=DataWire,
    label="Transcript",
    description="The text of a clip, with its language and timed segments. Never saved with the graph.",
    color="#b07ad9",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class TRANSCRIPT(PrimitiveType[Transcript | None]):
    """What a speech-to-text model heard: text, language and segments with start and end times."""

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self._value = _as_transcript(self._value)  # the type's default arrives as JSON

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"value": {}}``: a transcript is not serialized."""
        return {"value": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> None:
        """Return ``None``: a transcript never round-trips."""
        return None


class TRANSCRIPTField(PrimitiveField):
    """DataField for TRANSCRIPT: starts empty (``None``)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._default = _as_transcript(self._default)
        self._value = self._default

    def _set_own(self, value: Any) -> None:
        self._value = _as_transcript(value)


def _as_transcript(value: Any) -> Transcript | None:
    return value if isinstance(value, Transcript) else None


TRANSCRIPT.field_class = TRANSCRIPTField
