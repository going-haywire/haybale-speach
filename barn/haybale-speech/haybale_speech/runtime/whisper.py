"""Load a Whisper model with faster-whisper, and transcribe a clip with it."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from haybale_speech.value import Segment, SttModel, Transcript

#: The backend name in catalog variants and on the card.
BACKEND = "faster-whisper"


class Cancelled(Exception):
    """The run was stopped between two segments."""


def load_whisper(
    path: Path, device: str, compute_type: str, threads: int, *, name: str, notes: tuple[str, ...] = ()
) -> SttModel:
    """Load the CTranslate2 Whisper model in the folder *path*.

    faster-whisper is given the folder only, never a model name, so it never downloads.

    Args:
        path: A folder holding ``model.bin``, ``config.json``, ``tokenizer.json`` and the vocabulary.
        device: ``cpu`` or ``cuda``.
        compute_type: A CTranslate2 compute type (see ``resolve_compute_type``).
        threads: CPU threads; ``0`` lets CTranslate2 choose.
        name: The model's name for the card.
        notes: Notes from resolving the device and compute type, kept for the card.
    """
    from faster_whisper import WhisperModel

    if not (path / "model.bin").is_file():
        raise ValueError(f"{path.name} is not a CTranslate2 Whisper model folder (no model.bin)")
    engine = WhisperModel(
        str(path), device=device, compute_type=compute_type, cpu_threads=threads, local_files_only=True
    )
    return SttModel(
        engine=engine, backend=BACKEND, device=device, compute_type=compute_type, name=name, notes=notes
    )


def release_whisper(model: SttModel) -> None:
    """Nothing to free by hand: the model's memory goes with its last reference."""


def transcribe(
    model: SttModel,
    samples: np.ndarray,
    *,
    language: str | None = None,
    cancel: threading.Event | None = None,
    on_progress: Callable[[float, int], None] | None = None,
) -> Transcript:
    """Transcribe mono *samples* at the model's sample rate.

    Args:
        model: A loaded model.
        samples: 1-D float32 samples at ``model.sample_rate``.
        language: A language code; ``None`` detects it.
        cancel: Checked between segments; once set, :class:`Cancelled` is raised.
        on_progress: Called after each segment with the share of the clip done and the
            segments so far.

    Raises:
        Cancelled: *cancel* was set.
    """
    duration = len(samples) / model.sample_rate
    with model.lock:
        segments, info = model.engine.transcribe(samples, language=language or None)
        found = collect_segments(segments, duration, cancel=cancel, on_progress=on_progress)
    return Transcript.from_segments(found, str(info.language or language or ""), duration)


def collect_segments(
    segments: Iterable[Any],
    duration: float,
    *,
    cancel: threading.Event | None = None,
    on_progress: Callable[[float, int], None] | None = None,
) -> list[Segment]:
    """Turn faster-whisper's lazy segments into :class:`Segment` values, one at a time.

    Each segment is decoded as the loop asks for it, so *cancel* stops the work between two.
    """
    found: list[Segment] = []
    for raw in segments:
        if cancel is not None and cancel.is_set():
            raise Cancelled
        found.append(Segment(start=float(raw.start), end=float(raw.end), text=str(raw.text).strip()))
        if on_progress is not None:
            share = min(1.0, float(raw.end) / duration) if duration > 0 else 1.0
            on_progress(share, len(found))
    return found
