"""VOICE: how a text-to-speech model sounds, for Speak."""

from typing import Any

from haywire.api import DataWire, PrimitiveField, PrimitiveType, StoreStrategy, type as type_decorator

from haybale_speech.value.voice import Voice


@type_decorator(
    wire_type=DataWire,
    label="Voice",
    description="A voice a text-to-speech model speaks with. Never saved with the graph.",
    color="#d9a07a",
    store_strategy=StoreStrategy.NEVER,
    default={"value": {}},
)
class VOICE(PrimitiveType[Voice | None]):
    """A voice for Speak: its name, its language and the backend's voice data.

    Pick Voice gives one of a model's own voices. A voice runs only on a model of its backend.
    """

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self._value = _as_voice(self._value)  # the type's default arrives as JSON

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"value": {}}``: a voice is never serialized."""
        return {"value": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> None:
        """Return ``None``: a voice never round-trips."""
        return None


class VOICEField(PrimitiveField):
    """DataField for VOICE: a ``Voice``, or ``None`` until one is picked."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._default = _as_voice(self._default)
        self._value = self._default

    def _set_own(self, value: Any) -> None:
        self._value = _as_voice(value)


def _as_voice(value: Any) -> Voice | None:
    return value if isinstance(value, Voice) else None


VOICE.field_class = VOICEField
