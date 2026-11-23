"""SttModel — a speech-to-text model loaded on a device, ready to transcribe."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, eq=False)
class SttModel:
    """A loaded speech-to-text model, shared by every node that runs it.

    ``engine`` is the backend's own model object (a ``faster_whisper.WhisperModel``).
    Runs go through ``lock`` so two flows never drive one model at once.
    """

    engine: Any
    backend: str
    device: str
    compute_type: str
    name: str
    notes: tuple[str, ...] = ()
    lock: threading.Lock = field(default_factory=threading.Lock)

    #: Sample rate the model takes, in Hz.
    sample_rate: int = 16000

    def status(self) -> str:
        """One line for the card, e.g. ``faster-whisper-small · faster-whisper · cpu int8``."""
        return f"{self.name} · {self.backend} · {self.device} {self.compute_type}"
