"""Load a Kokoro model with kokoro-onnx, and speak text with it."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from haybale_speech.runtime.devices import NO_CUDA_DEVICE, resolve_device
from haybale_speech.runtime.whisper import Cancelled
from haybale_speech.value import TtsModel, Voice

#: The backend name in catalog variants and on the card.
BACKEND = "kokoro-onnx"

#: The model kind this backend loads.
KIND = "tts"

#: The module that must be importable for this backend to load.
REQUIRES = "kokoro_onnx"

#: Kokoro's sample rate, in Hz.
SAMPLE_RATE = 24000

#: Silence between two sentences, in seconds (kokoro-onnx's own sentence pause).
SENTENCE_PAUSE = 0.25

#: Slowest and fastest speed Kokoro takes.
SPEED_RANGE = (0.5, 2.0)

#: The language a voice speaks, by the first letter of its name (``af_heart`` is American English).
LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}

#: Splits text after the end of a sentence, or at a line break. Full-width marks (CJK text)
#: end a sentence with or without a space after them.
_SENTENCE_END = re.compile(r"(?<=[.!?…;:])\s+|(?<=[。！？；：])\s*|\n+")


def resolve(device: str, compute_type: str, cuda: bool | None = None) -> tuple[str, str, tuple[str, ...]]:
    """Return the device, the compute type (always ``""``) and notes, for the Device and Compute type settings.

    ONNX Runtime runs the model in the precision it was exported in, so Compute type does not
    apply; any choice but ``auto`` gets a note, and the card shows the variant's precision. CUDA needs an onnxruntime with the CUDA provider.

    Args:
        device: The Device setting.
        compute_type: The Compute type setting.
        cuda: Whether the CUDA provider is available; ``None`` asks ONNX Runtime.
    """
    if cuda is None:
        import onnxruntime

        cuda = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    resolved, device_note = resolve_device(device, lambda: None if cuda else NO_CUDA_DEVICE)
    notes = [device_note] if device_note else []
    wanted = (compute_type or "auto").strip().lower()
    if wanted != "auto":
        notes.append(f"compute type {wanted} does not apply to {BACKEND} — the model file sets it")
    return resolved, "", tuple(notes)


def model_files(path: Path) -> tuple[Path, Path]:
    """Return the ``.onnx`` model and the voices file in the folder *path*.

    Raises:
        ValueError: *path* is not a folder holding one ``.onnx`` file and a ``voices*.bin``.
    """
    if not path.is_dir():
        raise ValueError(f"{path.name} is not a Kokoro model folder (it needs an .onnx and a voices .bin)")
    models = sorted(path.glob("*.onnx"))
    voices = sorted(path.glob("voices*.bin"))
    if len(models) != 1 or len(voices) != 1:
        raise ValueError(
            f"{path.name} is not a Kokoro model folder (it needs one .onnx and one voices .bin)"
        )
    return models[0], voices[0]


def load(
    path: Path,
    device: str,
    compute_type: str,
    threads: int,
    *,
    name: str,
    notes: tuple[str, ...] = (),
    workers: int = 1,
) -> TtsModel:
    """Load the Kokoro model in the folder *path*.

    The ONNX Runtime session is built here from the resolved files, so kokoro-onnx never
    downloads and never picks the providers itself.

    Args:
        path: A folder holding the model's ``.onnx`` file and its ``voices*.bin``.
        device: ``cpu`` or ``cuda``.
        compute_type: The precision shown on the card, such as ``fp32``; ``""`` when unknown.
        threads: CPU threads; ``0`` lets ONNX Runtime choose.
        name: The model's name for the card.
        notes: Notes from resolving the settings, kept for the card.
        workers: Texts the model speaks at once. ONNX Runtime runs one session from several
            threads, and kokoro-onnx locks only its phonemizer (espeak-ng), so this is a limit only.
    """
    import onnxruntime
    from kokoro_onnx import Kokoro

    model_path, voices_path = model_files(path)
    options = onnxruntime.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    )
    session = onnxruntime.InferenceSession(str(model_path), sess_options=options, providers=providers)
    device, notes = ran_on(session, device, notes)
    engine = Kokoro.from_session(session, str(voices_path))
    return TtsModel(
        engine=engine,
        backend=BACKEND,
        device=device,
        compute_type=compute_type,
        name=name,
        voices=tuple(engine.get_voices()),
        sample_rate=SAMPLE_RATE,
        notes=notes,
        workers=max(1, workers),
    )


def ran_on(session: Any, device: str, notes: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """Return the device *session* really runs on, with a note when CUDA fell back to the CPU.

    ONNX Runtime lists the CUDA provider whenever its library is installed, but drops it
    without an error when CUDA cannot start; the session's first provider is the one it uses.
    """
    if device != "cuda":
        return device, notes
    first = session.get_providers()[0]
    if first == "CUDAExecutionProvider":
        return device, notes
    return "cpu", (*notes, f"cuda cannot run in ONNX Runtime ({first} only) — running on cpu")


def release(model: TtsModel) -> None:
    """Nothing to free by hand: the session's memory goes with its last reference."""


def language_of(voice: str) -> str:
    """Return the language a Kokoro voice speaks, from the first letter of its name; ``en-us`` if unknown."""
    return LANGUAGES.get(voice[:1].lower(), "en-us")


def voice(model: TtsModel, name: str) -> Voice:
    """Return the model's own voice *name*.

    Raises:
        ValueError: The model has no voice of that name; the message lists the ones it has.
    """
    wanted = (name or "").strip()
    if wanted not in model.voices:
        raise ValueError(f"{model.name} has no voice {wanted!r}; it has {', '.join(model.voices)}")
    return Voice(
        name=wanted, backend=model.backend, language=language_of(wanted), style=model.engine.voices[wanted]
    )


def sentences(text: str) -> list[str]:
    """Split *text* into sentences, dropping those with no letter or digit (``---``, ``…``)."""
    parts = (part.strip() for part in _SENTENCE_END.split(text) if part)
    return [part for part in parts if any(char.isalnum() for char in part)]


def speak(
    model: TtsModel,
    text: str,
    voice: Voice,
    *,
    speed: float = 1.0,
    cancel: threading.Event | None = None,
    on_progress: Callable[[float, int], None] | None = None,
) -> np.ndarray:
    """Speak *text* with *voice*, one sentence at a time, and return the mono float32 samples.

    Args:
        model: A loaded model.
        text: The text to speak.
        voice: A voice of the model's backend.
        speed: 0.5 to 2.0; 1.0 is the voice's own pace.
        cancel: Checked between sentences; once set, :class:`Cancelled` is raised.
        on_progress: Called after each sentence with the share of the sentences done and the
            number done.

    Raises:
        ValueError: The voice is of another backend, the speed is out of range, or there is no text.
        Cancelled: *cancel* was set.
    """
    if voice.backend != model.backend:
        raise ValueError(f"voice {voice.name} is a {voice.backend} voice; this model runs {model.backend}")
    low, high = SPEED_RANGE
    if not low <= speed <= high:
        raise ValueError(f"speed must be between {low} and {high}, got {speed}")
    parts = sentences(text)
    if not parts:
        raise ValueError("no text to speak")
    pause = np.zeros(int(SENTENCE_PAUSE * model.sample_rate), np.float32)
    pieces: list[np.ndarray] = []
    with model.slots:
        for index, sentence in enumerate(parts):
            if cancel is not None and cancel.is_set():
                raise Cancelled
            try:
                audio, _rate = model.engine.create(
                    sentence, voice=voice.style, speed=speed, lang=voice.language
                )
            except ValueError as error:
                if "Nothing to synthesize" not in str(error):
                    raise
                audio = None  # a piece with no phonemes is skipped, not the whole text
            if audio is not None:
                if pieces:
                    pieces.append(pause)
                pieces.append(np.asarray(audio, np.float32).ravel())
            if on_progress is not None:
                on_progress((index + 1) / len(parts), index + 1)
    if not pieces:
        raise ValueError("no text to speak")
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)
