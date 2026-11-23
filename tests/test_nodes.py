"""Load STT Model, Transcribe and Collect Clip in a graph, with a fake engine in place of faster-whisper."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from haybale_audio.value import AudioBlock
from haywire.core.volumes import VolumeStore

from haybale_speech.value import SttModel
from tests.conftest import add_node, place_model, run_from

pytestmark = pytest.mark.integration

LOAD = ("haybale_speech.nodes.load_stt_model", "LoadSttModelNode")
TRANSCRIBE = ("haybale_speech.nodes.transcribe", "TranscribeNode")
CLIP = ("haybale_speech.nodes.collect_clip", "CollectClipNode")


class _FakeEngine:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> Any:
        self.lengths.append(len(samples))
        segment = SimpleNamespace(start=0.0, end=len(samples) / 16000, text=" Hello there.")
        return iter([segment]), SimpleNamespace(language=language or "en")


@pytest.fixture
def fake_load(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Make Load STT Model build an SttModel around a fake engine; returns the load calls."""
    mod = importlib.import_module("haybale_speech.nodes.load_stt_model")
    calls: list[tuple[Any, ...]] = []

    def load(path: Path, device: str, compute_type: str, threads: int, **kwargs: Any) -> SttModel:
        calls.append((path, device, compute_type, threads))
        return SttModel(
            engine=_FakeEngine(),
            backend="faster-whisper",
            device=device,
            compute_type=compute_type,
            name=kwargs["name"],
            notes=kwargs.get("notes", ()),
        )

    monkeypatch.setattr(mod, "load_whisper", load)
    monkeypatch.setattr(mod, "resolve_device", lambda name: ("cpu", None))
    return calls


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "fake-whisper.bin"
    path.write_bytes(b"not a real model")
    return path


def _graph(graph: Any) -> tuple[Any, Any, Any, Any]:
    begin = add_node(graph, "haybale_core.nodes.events.begin_play_event", "BeginPlayEventNode")
    printer = add_node(graph, "haybale_core.nodes.print", "PrintNode")
    load = add_node(graph, *LOAD)
    transcribe = add_node(graph, *TRANSCRIBE)
    graph.create_edge_wrapper(begin.node_id, "exec", printer.node_id, "exec")
    graph.create_edge_wrapper(load.node_id, "loaded", transcribe.node_id, "model")
    graph.create_edge_wrapper(transcribe.node_id, "text", printer.node_id, "message")
    return begin, load, transcribe, printer


def _pool() -> Any:
    from haybale_models.consumer import models_state

    return models_state().pool


def test_a_clip_transcribes_through_the_loaded_model(
    graph: Any, models_env: VolumeStore, model_file: Path, fake_load: list[tuple[Any, ...]]
) -> None:
    begin, load, transcribe, _ = _graph(graph)
    load.ports["model"].set_value(place_model(model_file))
    stereo_48k = AudioBlock(np.zeros((2, 48000), np.float32), 48000)
    transcribe.ports["audio"].set_value(stereo_48k)
    transcribe.ports["language"].set_value("de")

    run_from(graph, begin)

    assert transcribe.ports["text"].get_value() == "Hello there."
    result = transcribe.ports["transcript"].get_value()
    assert result.language == "de"
    assert result.duration == pytest.approx(1.0)
    assert len(fake_load) == 1
    _path, device, compute_type, threads = fake_load[0]
    assert (device, compute_type, threads) == ("cpu", "int8", 0)
    assert load.status.loaded == "< shows after the first run >"  # released when the run ended
    assert _pool().entries() == []


def test_transcribe_without_a_model_is_an_error(graph: Any) -> None:
    transcribe = add_node(graph, *TRANSCRIBE)
    with pytest.raises(ValueError, match="no model"):
        transcribe.worker(SimpleNamespace(), audio=None, model=None, language="auto")


def _context(pin: str) -> Any:
    return SimpleNamespace(control_pin=pin)


def test_collect_clip_sends_the_gathered_blocks_on_stop(graph: Any) -> None:
    clip = add_node(graph, *CLIP)
    block = AudioBlock(np.full((1, 1600), 0.5, np.float32), 16000)

    assert clip.worker(_context("stop")) is None  # nothing gathered: nothing sent
    assert clip.worker(_context("add"), audio=block, max_seconds=300.0) == "next"
    assert clip.worker(_context("add"), audio=block, max_seconds=300.0) == "next"
    assert clip.ports["seconds"].get_value() == pytest.approx(0.2)
    assert clip.worker(_context("stop")) == "done"

    sent = clip.ports["clip"].get_value()
    assert (sent.channels, sent.frames, sent.sample_rate) == (1, 3200, 16000)
    assert clip.ports["status"].get_value() == "Sent 0.2 s"
    assert clip.worker(_context("stop")) is None  # the next clip starts empty


def test_collect_clip_drops_blocks_past_max_seconds(graph: Any) -> None:
    clip = add_node(graph, *CLIP)
    block = AudioBlock(np.zeros((1, 1600), np.float32), 16000)
    for _ in range(3):
        assert clip.worker(_context("add"), audio=block, max_seconds=0.25) == "next"
    assert clip.ports["status"].get_value() == "Full at 0.2 s — 1 blocks dropped"
    clip.worker(_context("stop"))
    assert clip.ports["clip"].get_value().frames == 3200
