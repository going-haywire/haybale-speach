"""TtsModel — a text-to-speech model loaded on a device, ready to speak."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, eq=False)
class TtsModel:
    """A loaded text-to-speech model, shared by every node that runs it.

    ``engine`` is the backend's own model object (a ``kokoro_onnx.Kokoro``). ``voices`` are
    the names of the voices it ships with. A run holds one of its ``workers`` ``slots``, so at
    most that many flows drive it at once.
    """

    engine: Any
    backend: str
    device: str
    compute_type: str
    name: str
    voices: tuple[str, ...] = ()
    sample_rate: int = 24000
    notes: tuple[str, ...] = ()
    #: How many runs may use the model at once (the loader's Workers).
    workers: int = 1
    slots: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", threading.BoundedSemaphore(max(1, self.workers)))

    def status(self) -> str:
        """One line for the card, e.g. ``Kokoro 82M · kokoro-onnx · cpu fp32``."""
        line = f"{self.name} · {self.backend} · {self.device} {self.compute_type}".rstrip()
        return f"{line} · {self.workers} workers" if self.workers > 1 else line
