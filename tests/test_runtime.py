"""The runtime pieces without a model: audio conversion, the clip buffer, devices, segments."""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
from haybale_audio.value import AudioBlock

from haybale_speech.runtime.audio import model_input
from haybale_speech.runtime.clip import ClipBuffer
from haybale_speech.runtime import devices
from haybale_speech.runtime.devices import NO_CUDA_DEVICE, cuda_problem, resolve_compute_type, resolve_device
from haybale_speech.runtime.whisper import Cancelled, collect_segments, transcribe
from haybale_speech.value import Segment, SttModel, Transcript

pytestmark = pytest.mark.unit


def _block(channels: int, frames: int, rate: int, value: float = 0.0) -> AudioBlock:
    return AudioBlock(np.full((channels, frames), value, np.float32), rate)


def test_model_input_mixes_stereo_to_mono() -> None:
    data = np.stack([np.full(160, 0.5, np.float32), np.full(160, -0.1, np.float32)])
    out = model_input(AudioBlock(data, 16000), 16000)
    assert out.shape == (160,)
    assert out.dtype == np.float32
    assert np.allclose(out, 0.2)


def test_model_input_resamples_the_whole_clip() -> None:
    rate = 48000
    tone = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)
    out = model_input(AudioBlock(tone[None, :], rate), 16000)
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 2  # one second in, one second out: no tail held back
    assert 0.6 < float(np.abs(out).max()) < 1.1


def test_clip_buffer_joins_blocks_and_starts_over() -> None:
    buffer = ClipBuffer(max_seconds=10)
    assert buffer.take() is None
    assert buffer.add(_block(2, 100, 1000, 0.1))
    assert buffer.add(_block(2, 50, 1000, 0.2))
    assert buffer.seconds == pytest.approx(0.15)
    clip = buffer.take()
    assert clip is not None
    assert (clip.channels, clip.frames, clip.sample_rate) == (2, 150, 1000)
    assert np.allclose(clip.data[:, 100:], 0.2)
    assert buffer.take() is None
    assert buffer.add(_block(1, 10, 8000))  # a new clip takes a new format


def test_clip_buffer_refuses_another_format() -> None:
    buffer = ClipBuffer(max_seconds=10)
    buffer.add(_block(1, 100, 16000))
    with pytest.raises(ValueError, match="1 ch at 16000 Hz"):
        buffer.add(_block(1, 100, 48000))


def test_clip_buffer_drops_blocks_past_the_limit() -> None:
    buffer = ClipBuffer(max_seconds=0.25)
    assert buffer.add(_block(1, 200, 1000))
    assert not buffer.add(_block(1, 100, 1000))
    assert buffer.dropped == 1
    clip = buffer.take()
    assert clip is not None and clip.frames == 200
    assert buffer.dropped == 0


BROKEN = "libcublas.so.12 is not found (cuBLAS for CUDA 12)"


@pytest.mark.parametrize(
    ("name", "problem", "expected"),
    [
        ("auto", NO_CUDA_DEVICE, ("cpu", None)),
        ("auto", None, ("cuda", None)),
        ("auto", BROKEN, ("cpu", f"cuda cannot run: {BROKEN} — running on cpu")),
        ("", NO_CUDA_DEVICE, ("cpu", None)),
        ("", BROKEN, ("cpu", f"cuda cannot run: {BROKEN} — running on cpu")),
        ("cpu", None, ("cpu", None)),
        ("cuda", None, ("cuda", None)),
        ("cuda", NO_CUDA_DEVICE, ("cpu", "cuda not available — running on cpu")),
        ("mps", NO_CUDA_DEVICE, ("cpu", "mps is not a device — running on cpu")),
    ],
)
def test_resolve_device(name: str, problem: str | None, expected: tuple[str, str | None]) -> None:
    assert resolve_device(name, lambda: problem) == expected


def test_resolve_device_explicit_cuda_that_cannot_run_fails_with_the_reason() -> None:
    with pytest.raises(RuntimeError, match="libcublas.so.12 is not found"):
        resolve_device("cuda", lambda: BROKEN)


def test_resolve_device_cpu_never_probes() -> None:
    def probe() -> str | None:
        raise AssertionError("probed")

    assert resolve_device("cpu", probe) == ("cpu", None)


@pytest.fixture
def fresh_probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """cuda_problem with a fake CTranslate2 and cuBLAS loader, its cache cleared around the test."""
    state = SimpleNamespace(devices=2, loads=True, counts=0)

    def count() -> int:
        state.counts += 1
        return state.devices

    def load(library: str) -> object:
        if not state.loads:
            raise OSError(f"{library}: cannot open shared object file")
        return object()

    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=count))
    monkeypatch.setattr(devices.ctypes, "CDLL", load)
    monkeypatch.setattr(devices.sys, "platform", "linux")
    cuda_problem.cache_clear()
    yield state
    cuda_problem.cache_clear()


def test_cuda_problem_present_and_working(fresh_probe: SimpleNamespace) -> None:
    assert cuda_problem() is None


def test_cuda_problem_present_but_broken(fresh_probe: SimpleNamespace) -> None:
    fresh_probe.loads = False
    assert cuda_problem() == "libcublas.so.12 is not found (cuBLAS for CUDA 12)"


def test_cuda_problem_absent(fresh_probe: SimpleNamespace) -> None:
    fresh_probe.devices = 0
    assert cuda_problem() == NO_CUDA_DEVICE


def test_cuda_problem_is_asked_once(fresh_probe: SimpleNamespace) -> None:
    cuda_problem()
    cuda_problem()
    assert fresh_probe.counts == 1


CPU = {"int8", "int8_float32", "float32"}
CUDA = {"int8", "int8_float16", "float16", "bfloat16", "float32"}


@pytest.mark.parametrize(
    ("name", "device", "supported", "expected"),
    [
        ("auto", "cpu", CPU, ("int8", None)),
        ("auto", "cuda", CUDA, ("float16", None)),
        ("auto", "cpu", {"float32"}, ("float32", None)),
        ("int8_float32", "cpu", CPU, ("int8_float32", None)),
        ("float16", "cpu", CPU, ("float32", "float16 not supported on cpu — running float32")),
        ("fp8", "cpu", CPU, ("float32", "fp8 is not a compute type — running float32")),
    ],
)
def test_resolve_compute_type(
    name: str, device: str, supported: set[str], expected: tuple[str, str | None]
) -> None:
    assert resolve_compute_type(name, device, supported) == expected


def _raw(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text)


def test_collect_segments_maps_and_reports_progress() -> None:
    progress: list[tuple[float, int]] = []
    raw = [_raw(0.0, 1.5, " Hello"), _raw(1.5, 4.0, " world. ")]
    found = collect_segments(raw, 4.0, on_progress=lambda share, n: progress.append((share, n)))
    assert found == [Segment(0.0, 1.5, "Hello"), Segment(1.5, 4.0, "world.")]
    assert progress == [(0.375, 1), (1.0, 2)]


def test_collect_segments_stops_between_segments() -> None:
    cancel = threading.Event()
    pulled: list[int] = []

    def lazy():  # faster-whisper decodes each segment as the loop asks for it
        for i in range(5):
            pulled.append(i)
            if i == 1:
                cancel.set()
            yield _raw(i, i + 1, f"s{i}")

    with pytest.raises(Cancelled):
        collect_segments(lazy(), 5.0, cancel=cancel)
    assert pulled == [0, 1]


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None]] = []

    def transcribe(self, samples: np.ndarray, language: str | None = None):
        self.calls.append((len(samples), language))
        segments = iter([_raw(0.0, 0.5, " one"), _raw(0.5, 1.0, " two")])
        return segments, SimpleNamespace(language=language or "en")


def _model(engine: object) -> SttModel:
    return SttModel(engine=engine, backend="fake", device="cpu", compute_type="int8", name="fake")


def test_transcribe_joins_segments_into_a_transcript() -> None:
    engine = _FakeEngine()
    result = transcribe(_model(engine), np.zeros(16000, np.float32), language="de")
    assert result == Transcript(
        text="one two",
        language="de",
        segments=(Segment(0.0, 0.5, "one"), Segment(0.5, 1.0, "two")),
        duration=1.0,
    )
    assert engine.calls == [(16000, "de")]


def test_transcribe_detects_the_language_when_none_is_given() -> None:
    result = transcribe(_model(_FakeEngine()), np.zeros(8000, np.float32))
    assert result.language == "en"
    assert result.duration == 0.5
