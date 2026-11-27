"""SttModel — a speech-to-text model loaded on a device, ready to transcribe."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, eq=False)
class SttModel:
    """A loaded speech-to-text model, shared by every node that runs it.

    ``engine`` is the backend's own model object (a ``faster_whisper.WhisperModel``).
    A run holds one of its ``workers`` ``slots``, so at most that many flows drive it at once.
    """

    engine: Any
    backend: str
    device: str
    compute_type: str
    name: str
    notes: tuple[str, ...] = ()
    #: How many runs may use the model at once (the loader's Workers).
    workers: int = 1
    slots: threading.BoundedSemaphore = field(init=False, repr=False)

    #: Sample rate the model takes, in Hz.
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", threading.BoundedSemaphore(max(1, self.workers)))

    def status(self) -> str:
        """One line for the card, e.g. ``faster-whisper-small · faster-whisper · cpu int8``."""
        line = f"{self.name} · {self.backend} · {self.device} {self.compute_type}"
        return f"{line} · {self.workers} workers" if self.workers > 1 else line
