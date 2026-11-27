"""The speech nodes in a graph, with fake engines in place of faster-whisper and kokoro-onnx."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from haybale_audio.value import AudioBlock
from haywire.core.volumes import VolumeStore

from haybale_speech.runtime.whisper import resolve as WHISPER_RESOLVE
from haybale_speech.value import SttModel, TtsModel
from tests.conftest import add_node, place_model, run_from

pytestmark = pytest.mark.integration

LOAD = ("haybale_speech.nodes.load_speech_model", "LoadSpeechModelNode")
LOAD_STT = ("haybale_speech.nodes.load_stt_model", "LoadSttModelNode")
TRANSCRIBE = ("haybale_speech.nodes.transcribe", "TranscribeNode")
SPEAK = ("haybale_speech.nodes.speak", "SpeakNode")
PICK = ("haybale_speech.nodes.pick_voice", "PickVoiceNode")
CLIP = ("haybale_speech.nodes.collect_clip", "CollectClipNode")


class _FakeEngine:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def transcribe(self, samples: np.ndarray, language: str | None = None) -> Any:
        self.lengths.append(len(samples))
        segment = SimpleNamespace(start=0.0, end=len(samples) / 16000, text=" Hello there.")
        return iter([segment]), SimpleNamespace(language=language or "en")


class _FakeKokoro:
    """Speaks 0.1 s of a constant per sentence: the voice's first style value."""

    def __init__(self) -> None:
        self.voices = {
            "af_heart": np.full((510, 1, 256), 0.5, np.float32),
            "bf_emma": np.zeros((510, 1, 256)),
        }
        self.calls: list[tuple[str, float, str]] = []

    def create(self, text: str, voice: Any, speed: float, lang: str) -> tuple[np.ndarray, int]:
        self.calls.append((text, speed, lang))
        return np.full(2400, float(np.asarray(voice).flat[0]), np.float32), 24000


@pytest.fixture
def fake_load(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Make the faster-whisper backend build an SttModel around a fake engine; returns the load calls."""
    whisper = importlib.import_module("haybale_speech.runtime.whisper")
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

    monkeypatch.setattr(whisper, "load", load)
    monkeypatch.setattr(whisper, "resolve", lambda device, compute: ("cpu", "int8", ()))
    return calls


@pytest.fixture
def fake_kokoro(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Make the kokoro-onnx backend build a TtsModel around a fake engine; returns the load calls."""
    kokoro = importlib.import_module("haybale_speech.runtime.kokoro")
    calls: list[tuple[Any, ...]] = []

    def load(path: Path, device: str, compute_type: str, threads: int, **kwargs: Any) -> TtsModel:
        calls.append((path, device, compute_type, threads))
        engine = _FakeKokoro()
        return TtsModel(
            engine=engine,
            backend="kokoro-onnx",
            device=device,
            compute_type=compute_type,
            name=kwargs["name"],
            voices=tuple(engine.voices),
            notes=kwargs.get("notes", ()),
        )

    monkeypatch.setattr(kokoro, "load", load)
    monkeypatch.setattr(kokoro, "resolve", lambda device, compute: ("cpu", "", ()))
    return calls


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "fake-whisper.bin"
    path.write_bytes(b"not a real model")
    return path


def _graph(graph: Any, loader: tuple[str, str] = LOAD) -> tuple[Any, Any, Any, Any]:
    begin = add_node(graph, "haybale_core.nodes.events.begin_play_event", "BeginPlayEventNode")
    printer = add_node(graph, "haybale_core.nodes.print", "PrintNode")
    load = add_node(graph, *loader)
    transcribe = add_node(graph, *TRANSCRIBE)
    graph.create_edge_wrapper(begin.node_id, "exec", printer.node_id, "exec")
    graph.create_edge_wrapper(load.node_id, "loaded", transcribe.node_id, "model")
    graph.create_edge_wrapper(transcribe.node_id, "text", printer.node_id, "message")
    return begin, load, transcribe, printer


def _loader_for(graph: Any, consumer: tuple[str, str], port: str = "model") -> tuple[Any, Any]:
    """A Load Speech Model with its outlet wired to a new *consumer*, which sets its kind."""
    load = add_node(graph, *LOAD)
    node = add_node(graph, *consumer)
    graph.create_edge_wrapper(load.node_id, "loaded", node.node_id, port)
    return load, node


def _pool() -> Any:
    from haybale_models.consumer import models_state

    return models_state().pool


@pytest.mark.parametrize("loader", [LOAD, LOAD_STT], ids=["load-speech-model", "load-stt-model"])
def test_a_clip_transcribes_through_the_loaded_model(
    graph: Any,
    models_env: VolumeStore,
    model_file: Path,
    fake_load: list[tuple[Any, ...]],
    loader: tuple[str, str],
) -> None:
    begin, load, transcribe, _ = _graph(graph, loader)
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


def test_the_outlet_takes_the_kind_of_the_node_it_feeds(graph: Any) -> None:
    from haybale_speech.types.stt_model import STT_MODEL
    from haybale_speech.types.tts_model import TTS_MODEL

    open_loader = add_node(graph, *LOAD)
    assert "model" not in open_loader.ports  # the picker appears once the outlet has a kind

    for consumer, kind, purpose in ((TRANSCRIBE, STT_MODEL, "stt"), (SPEAK, TTS_MODEL, "tts")):
        load, _ = _loader_for(graph, consumer)
        assert load.ports["loaded"].type_cls is kind
        properties = load.ports["model"].widget_config["properties"]
        assert properties["filter"] == {"purpose": [purpose]}
        assert properties["prefer"][0] == {
            "backend": "faster-whisper" if purpose == "stt" else "kokoro-onnx"
        }


def test_an_open_outlet_is_an_error(graph: Any) -> None:
    load = add_node(graph, *LOAD)
    with pytest.raises(ValueError, match="connect the Model outlet first"):
        load.worker(MagicMock(), model=None)


def test_a_missing_backend_is_an_error_on_the_card(
    graph: Any, models_env: VolumeStore, model_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from haybale_speech.runtime.backends import Backend, BackendMissing

    monkeypatch.setattr(Backend, "installed", lambda self: False)
    load, _ = _loader_for(graph, SPEAK)
    load.ports["model"].set_value(place_model(model_file))
    with pytest.raises(BackendMissing, match="kokoro-onnx is not installed"):
        load._execute(MagicMock())
    assert _pool().entries() == []


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("auto", "cuda cannot run: libcublas.so.12 is not found (cuBLAS for CUDA 12) — running on cpu"),
        ("cuda", None),
    ],
)
def test_a_gpu_that_cannot_run_falls_back_on_auto_and_fails_on_cuda(
    graph: Any,
    models_env: VolumeStore,
    model_file: Path,
    fake_load: list[tuple[Any, ...]],
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    expected: str | None,
) -> None:
    whisper = importlib.import_module("haybale_speech.runtime.whisper")
    devices = importlib.import_module("haybale_speech.runtime.devices")
    broken = "libcublas.so.12 is not found (cuBLAS for CUDA 12)"
    monkeypatch.setattr(whisper, "resolve", WHISPER_RESOLVE)  # the real one, under the fake load
    monkeypatch.setattr(whisper, "resolve_device", lambda name: devices.resolve_device(name, lambda: broken))
    monkeypatch.setattr(whisper, "resolve_compute_type", lambda name, dev: ("int8", None))
    load, _ = _loader_for(graph, TRANSCRIBE)
    load.settings.device = device
    load.ports["model"].set_value(place_model(model_file))
    if expected is None:
        with pytest.raises(RuntimeError, match="pick cpu or auto as the Device"):
            load._execute(MagicMock())
        assert fake_load == []
        return
    load._execute(MagicMock())
    assert fake_load[0][1] == "cpu"
    assert [w.message for w in load.wrapper.state.warnings] == [expected]


def _loaded_tts(
    graph: Any, model_file: Path, consumer: tuple[str, str] = SPEAK
) -> tuple[Any, Any, TtsModel]:
    load, node = _loader_for(graph, consumer)
    load.ports["model"].set_value(place_model(model_file))
    load._execute(MagicMock())
    model = load.ports["loaded"].get_value()
    assert isinstance(model, TtsModel)
    return load, node, model


def test_speak_turns_text_into_a_mono_clip(
    graph: Any, models_env: VolumeStore, model_file: Path, fake_kokoro: list[tuple[Any, ...]]
) -> None:
    load, speak, model = _loaded_tts(graph, model_file)
    assert load.status.loaded == "fake-whisper.bin · kokoro-onnx · cpu"

    speak.worker(MagicMock(cancel=None), "Hello. How are you?", model, None, "af_heart", 1.25)

    clip = speak.ports["audio"].get_value()
    assert (clip.channels, clip.sample_rate) == (1, 24000)
    assert clip.frames == 2400 + 6000 + 2400  # two sentences and the pause between them
    assert speak.ports["seconds"].get_value() == pytest.approx(10800 / 24000)
    assert model.engine.calls == [("Hello.", 1.25, "en-us"), ("How are you?", 1.25, "en-us")]
    load.on_shutdown(MagicMock())
    assert _pool().entries() == []


def test_speak_names_the_voices_of_the_model_for_an_unknown_one(
    graph: Any, models_env: VolumeStore, model_file: Path, fake_kokoro: list[tuple[Any, ...]]
) -> None:
    load, speak, model = _loaded_tts(graph, model_file)
    with pytest.raises(ValueError, match="has no voice 'zz_nobody'; it has af_heart, bf_emma"):
        speak.worker(MagicMock(cancel=None), "Hello.", model, None, "zz_nobody", 1.0)
    load.on_shutdown(MagicMock())


def test_a_picked_voice_takes_the_place_of_the_voice_name(
    graph: Any, models_env: VolumeStore, model_file: Path, fake_kokoro: list[tuple[Any, ...]]
) -> None:
    load, pick, model = _loaded_tts(graph, model_file, PICK)
    pick.worker(MagicMock(), model, "bf_emma")
    voice = pick.ports["voice"].get_value()
    assert (voice.name, voice.language, voice.backend) == ("bf_emma", "en-gb", "kokoro-onnx")
    assert pick.ports["language"].get_value() == "en-gb"

    speak = add_node(graph, *SPEAK)
    speak.worker(MagicMock(cancel=None), "Cheerio.", model, voice, "af_heart", 1.0)
    assert model.engine.calls == [("Cheerio.", 1.0, "en-gb")]
    assert float(speak.ports["audio"].get_value().data.max()) == 0.0  # bf_emma's style, not af_heart's
    load.on_shutdown(MagicMock())


def test_speak_without_a_model_is_an_error(graph: Any) -> None:
    speak = add_node(graph, *SPEAK)
    with pytest.raises(ValueError, match="no model"):
        speak.worker(MagicMock(cancel=None), "Hello.", None, None, "af_heart", 1.0)


def test_a_tts_model_does_not_fit_transcribe(
    graph: Any, models_env: VolumeStore, model_file: Path, fake_kokoro: list[tuple[Any, ...]]
) -> None:
    """A catalog variant of the wrong kind is refused before it loads."""
    from haybale_speech.nodes.load_speech_model import _backend_for
    from haybale_speech.types.stt_model import STT_MODEL

    resolved = SimpleNamespace(resource=SimpleNamespace(backend="kokoro-onnx"))
    with pytest.raises(ValueError, match="is a tts model, which does not fit this outlet"):
        _backend_for(resolved, STT_MODEL, "Kokoro 82M")
    assert fake_kokoro == []


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
