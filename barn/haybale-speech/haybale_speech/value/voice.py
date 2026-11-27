"""Voice — how a text-to-speech model sounds: a named style for one backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, eq=False)
class Voice:
    """A voice a text-to-speech model speaks with.

    ``style`` is the backend's own voice data (for Kokoro a ``(510, 1, 256)`` float32 array of
    style vectors). ``language`` is the code the backend phonemizes the text in, such as
    ``en-us``. A voice runs only on a model of the same ``backend``.
    """

    name: str
    backend: str
    language: str
    style: Any
