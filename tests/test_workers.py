"""Workers: a loaded model runs at most that many calls at once, with fake engines that count them."""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from haybale_speech.runtime import kokoro, whisper
from haybale_speech.value import SttModel, TtsModel, Voice

pytestmark = pytest.mark.unit

CALLS = 4


class _Gauge:
    """Counts the calls inside the engine at once; each call stays a moment so they can overlap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.now = 0
        self.most = 0

    def run(self) -> None:
        with self._lock:
            self.now += 1
            self.most = max(self.most, self.now)
        time.sleep(0.05)
        with self._lock:
            self.now -= 1


class _Whisper:
    def __init__(self, gauge: _Gauge) -> None:
        self.gauge = gauge

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> Any:
        def segments() -> Any:  # faster-whisper decodes lazily, while the caller iterates
            self.gauge.run()
            yield SimpleNamespace(start=0.0, end=1.0, text=" hi")

        return segments(), SimpleNamespace(language="en")


class _Kokoro:
    def __init__(self, gauge: _Gauge) -> None:
        self.gauge = gauge

    def create(self, text: str, voice: Any, speed: float, lang: str) -> tuple[np.ndarray, int]:
        self.gauge.run()
        return np.ones(240, np.float32), 24000


def _all_at_once(call: Any) -> None:
    threads = [threading.Thread(target=call) for _ in range(CALLS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_transcribe_runs_as_many_clips_at_once_as_the_model_has_workers(workers: int) -> None:
    gauge = _Gauge()
    model = SttModel(
        engine=_Whisper(gauge),
        backend="faster-whisper",
        device="cpu",
        compute_type="int8",
        name="W",
        workers=workers,
    )
    samples = np.zeros(16000, np.float32)
    _all_at_once(lambda: whisper.transcribe(model, samples, language="en"))
    assert gauge.most == workers


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_speak_runs_as_many_texts_at_once_as_the_model_has_workers(workers: int) -> None:
    gauge = _Gauge()
    model = TtsModel(
        engine=_Kokoro(gauge),
        backend="kokoro-onnx",
        device="cpu",
        compute_type="",
        name="K",
        voices=("af_heart",),
        workers=workers,
    )
    voice = Voice(name="af_heart", backend="kokoro-onnx", language="en-us", style=np.ones((510, 1, 256)))
    _all_at_once(lambda: kokoro.speak(model, "Hello.", voice))
    assert gauge.most == workers


def test_the_card_names_the_workers_only_when_there_are_several() -> None:
    one = SttModel(engine=None, backend="faster-whisper", device="cpu", compute_type="int8", name="W")
    two = SttModel(
        engine=None, backend="faster-whisper", device="cpu", compute_type="int8", name="W", workers=2
    )
    assert one.status() == "W · faster-whisper · cpu int8"
    assert two.status() == "W · faster-whisper · cpu int8 · 2 workers"


def test_whisper_gets_the_workers_as_num_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    made: dict[str, Any] = {}

    class _WhisperModel:
        def __init__(self, path: str, **kwargs: Any) -> None:
            made.update(kwargs)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=_WhisperModel))
    (tmp_path / "model.bin").write_bytes(b"")
    model = whisper.load_whisper(tmp_path, "cpu", "int8", 2, name="W", workers=3)
    assert made["num_workers"] == 3 and made["cpu_threads"] == 2
    assert model.workers == 3
