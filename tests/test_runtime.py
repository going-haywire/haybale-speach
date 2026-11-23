"""The runtime pieces without a model: audio conversion, the clip buffer, devices, segments."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from haybale_audio.value import AudioBlock

from haybale_speech.runtime.audio import model_input
from haybale_speech.runtime.clip import ClipBuffer
from haybale_speech.runtime.devices import resolve_compute_type, resolve_device
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


@pytest.mark.parametrize(
    ("name", "cuda", "expected"),
    [
        ("auto", 0, ("cpu", None)),
        ("auto", 1, ("cuda", None)),
        ("", 0, ("cpu", None)),
        ("cpu", 1, ("cpu", None)),
        ("cuda", 1, ("cuda", None)),
        ("cuda", 0, ("cpu", "cuda not available — running on cpu")),
        ("mps", 0, ("cpu", "mps is not a device — running on cpu")),
    ],
)
def test_resolve_device(name: str, cuda: int, expected: tuple[str, str | None]) -> None:
    assert resolve_device(name, cuda) == expected


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
