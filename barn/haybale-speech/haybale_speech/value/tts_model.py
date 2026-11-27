"""TtsModel — a text-to-speech model loaded on a device, ready to speak."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, eq=False)
class TtsModel:
    """A loaded text-to-speech model, shared by every node that runs it.

    ``engine`` is the backend's own model object (a ``kokoro_onnx.Kokoro``). ``voices`` are
    the names of the voices it ships with. Runs go through ``lock`` so two flows never drive
    one model at once.
    """

    engine: Any
    backend: str
    device: str
    compute_type: str
    name: str
    voices: tuple[str, ...] = ()
    sample_rate: int = 24000
    notes: tuple[str, ...] = ()
    lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> str:
        """One line for the card, e.g. ``Kokoro 82M · kokoro-onnx · cpu fp32``."""
        return f"{self.name} · {self.backend} · {self.device} {self.compute_type}".rstrip()
