"""Text-to-speech runtime and the backend table, with a fake Kokoro engine."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from haybale_speech.runtime import backends, kokoro
from haybale_speech.runtime.whisper import Cancelled
from haybale_speech.value import TtsModel, Voice

pytestmark = pytest.mark.unit


class _Engine:
    def __init__(self, on_create: Any = None) -> None:
        self.voices = {"af_heart": np.ones((510, 1, 256), np.float32)}
        self.texts: list[str] = []
        self.on_create = on_create

    def create(self, text: str, voice: Any, speed: float, lang: str) -> tuple[np.ndarray, int]:
        self.texts.append(text)
        if self.on_create is not None:
            self.on_create()
        return np.ones(240, np.float32), 24000


def _model(engine: Any = None) -> TtsModel:
    return TtsModel(
        engine=engine or _Engine(),
        backend="kokoro-onnx",
        device="cpu",
        compute_type="",
        name="K",
        voices=("af_heart",),
    )


def _voice(backend: str = "kokoro-onnx") -> Voice:
    return Voice(
        name="af_heart", backend=backend, language="en-us", style=np.ones((510, 1, 256), np.float32)
    )


@pytest.mark.parametrize(
    ("device", "compute", "cuda", "expected"),
    [
        ("auto", "auto", False, ("cpu", "", ())),
        ("auto", "auto", True, ("cuda", "", ())),
        ("cuda", "auto", False, ("cpu", "", ("cuda not available — running on cpu",))),
        (
            "cpu",
            "int8",
            False,
            ("cpu", "", ("compute type int8 does not apply to kokoro-onnx — the model file sets it",)),
        ),
    ],
)
def test_resolve(device: str, compute: str, cuda: bool, expected: tuple[Any, ...]) -> None:
    assert kokoro.resolve(device, compute, cuda=cuda) == expected


def test_model_files_finds_the_model_and_the_voices(tmp_path: Path) -> None:
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"")
    assert kokoro.model_files(tmp_path) == (tmp_path / "kokoro-v1.0.onnx", tmp_path / "voices-v1.0.bin")


def test_model_files_refuses_a_single_file(tmp_path: Path) -> None:
    path = tmp_path / "kokoro.onnx"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="not a Kokoro model folder"):
        kokoro.model_files(path)
    with pytest.raises(ValueError, match="one .onnx and one voices .bin"):
        kokoro.model_files(tmp_path)  # an .onnx but no voices


def test_sentences_split_at_ends_and_line_breaks() -> None:
    assert kokoro.sentences("Hi there. How are you?\nFine!  ") == ["Hi there.", "How are you?", "Fine!"]
    assert kokoro.sentences("  \n ") == []


def test_sentences_drop_pieces_with_no_letter_or_digit() -> None:
    assert kokoro.sentences("Step 1. Mix it.\n---\nDone!") == ["Step 1.", "Mix it.", "Done!"]
    assert kokoro.sentences("--- ... !") == []


def test_sentences_split_after_full_width_marks_without_a_space() -> None:
    assert kokoro.sentences("你好。今天很好！再见？好的；是：对") == [
        "你好。",
        "今天很好！",
        "再见？",
        "好的；",
        "是：",
        "对",
    ]


def test_speak_skips_a_piece_with_no_text_and_speaks_the_rest() -> None:
    engine = _Engine()
    samples = kokoro.speak(_model(engine), "Step 1. Mix it.\n---\nDone!", _voice())
    assert engine.texts == ["Step 1.", "Mix it.", "Done!"]
    assert len(samples) == 3 * 240 + 2 * int(kokoro.SENTENCE_PAUSE * 24000)


class _NoPhonemes(_Engine):
    """An engine for which some pieces have no phonemes, as kokoro-onnx says for them."""

    def __init__(self, silent: set[str]) -> None:
        super().__init__()
        self.silent = silent

    def create(self, text: str, voice: Any, speed: float, lang: str) -> tuple[np.ndarray, int]:
        if text in self.silent:
            self.texts.append(text)
            raise ValueError(f"Nothing to synthesize, {text!r} produced no phonemes")
        return super().create(text, voice, speed, lang)


def test_speak_skips_a_piece_the_engine_finds_no_phonemes_in() -> None:
    steps: list[int] = []
    engine = _NoPhonemes({"Hm2."})
    samples = kokoro.speak(
        _model(engine), "One. Hm2. Two.", _voice(), on_progress=lambda s, n: steps.append(n)
    )
    assert engine.texts == ["One.", "Hm2.", "Two."]
    assert len(samples) == 2 * 240 + int(kokoro.SENTENCE_PAUSE * 24000)
    assert steps == [1, 2, 3]


def test_speak_with_no_phonemes_anywhere_is_no_text_to_speak() -> None:
    with pytest.raises(ValueError, match="no text to speak"):
        kokoro.speak(_model(_NoPhonemes({"Hm2.", "Hm3."})), "Hm2. Hm3.", _voice())


class _Session:
    def __init__(self, *providers: str) -> None:
        self.providers = list(providers)

    def get_providers(self) -> list[str]:
        return self.providers


@pytest.mark.parametrize(
    ("device", "providers", "expected"),
    [
        ("cpu", ("CPUExecutionProvider",), ("cpu", ("n",))),
        ("cuda", ("CUDAExecutionProvider", "CPUExecutionProvider"), ("cuda", ("n",))),
        (
            "cuda",
            ("CPUExecutionProvider",),
            ("cpu", ("n", "cuda cannot run in ONNX Runtime (CPUExecutionProvider only) — running on cpu")),
        ),
    ],
)
def test_ran_on_reads_the_device_from_the_session(
    device: str, providers: tuple[str, ...], expected: tuple[str, tuple[str, ...]]
) -> None:
    assert kokoro.ran_on(_Session(*providers), device, ("n",)) == expected


@pytest.mark.parametrize(
    ("name", "language"),
    [
        ("af_heart", "en-us"),
        ("bf_emma", "en-gb"),
        ("ff_siwis", "fr-fr"),
        ("zm_yunxi", "cmn"),
        ("xx_new", "en-us"),
    ],
)
def test_language_of_a_voice(name: str, language: str) -> None:
    assert kokoro.language_of(name) == language


def test_speak_joins_sentences_with_a_pause_and_reports_progress() -> None:
    steps: list[tuple[float, int]] = []
    samples = kokoro.speak(
        _model(), "One. Two. Three.", _voice(), on_progress=lambda s, n: steps.append((s, n))
    )
    pause = int(kokoro.SENTENCE_PAUSE * 24000)
    assert samples.dtype == np.float32 and samples.ndim == 1
    assert len(samples) == 3 * 240 + 2 * pause
    assert steps == [(pytest.approx(1 / 3), 1), (pytest.approx(2 / 3), 2), (1.0, 3)]


def test_speak_stops_between_sentences() -> None:
    cancel = threading.Event()
    engine = _Engine(on_create=cancel.set)
    with pytest.raises(Cancelled):
        kokoro.speak(_model(engine), "One. Two.", _voice(), cancel=cancel)
    assert engine.texts == ["One."]


@pytest.mark.parametrize(
    ("text", "voice", "speed", "match"),
    [
        ("Hi.", _voice("other"), 1.0, "is a other voice; this model runs kokoro-onnx"),
        ("Hi.", _voice(), 2.5, "speed must be between 0.5 and 2.0"),
        ("   ", _voice(), 1.0, "no text to speak"),
    ],
)
def test_speak_refuses(text: str, voice: Voice, speed: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        kokoro.speak(_model(), text, voice, speed=speed)


def test_voice_takes_the_model_style() -> None:
    model = _model()
    picked = kokoro.voice(model, "af_heart")
    assert picked.style is model.engine.voices["af_heart"]
    assert (picked.backend, picked.language) == ("kokoro-onnx", "en-us")


def test_the_backend_table() -> None:
    assert [b.name for b in backends.of_kind("stt")] == ["faster-whisper"]
    assert [b.name for b in backends.of_kind("tts")] == ["kokoro-onnx"]
    assert backends.default_for("tts").name == "kokoro-onnx"
    assert backends.default_for(None) is None
    assert backends.preference("tts") == [{"backend": "kokoro-onnx"}, {}]
    assert backends.by_name("faster-whisper").runtime().BACKEND == "faster-whisper"
    with pytest.raises(backends.BackendMissing, match="no speech backend is called nope"):
        backends.by_name("nope")


def test_every_backend_module_has_the_interface() -> None:
    for backend in backends.BACKENDS:
        module = backend.runtime()
        assert (module.BACKEND, module.KIND, module.REQUIRES) == (
            backend.name,
            backend.kind,
            backend.requires,
        )
        for name in ("resolve", "load", "release"):
            assert callable(getattr(module, name))


def test_preference_leaves_out_a_backend_that_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends.Backend, "installed", lambda self: self.name != "kokoro-onnx")
    assert backends.preference("tts") == [{}]
    assert backends.preference(None) == [{"backend": "faster-whisper"}, {}]
