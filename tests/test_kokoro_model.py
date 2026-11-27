"""The real Kokoro 82M speaks, and Whisper small hears what it said: the whole circle.

Needs both models in the Hugging Face cache (about 350 MB and 490 MB), so it runs only with
``SPEECH_MODEL_TESTS=1``; with that set, a missing snapshot is downloaded first.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

from haybale_speech.runtime import kokoro
from haybale_speech.runtime.audio import model_input
from haybale_speech.runtime.whisper import load_whisper, transcribe
from tests.conftest import REAL_HOME
from tests.test_whisper_model import whisper_small  # noqa: F401  (a fixture)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get("SPEECH_MODEL_TESTS"), reason="set SPEECH_MODEL_TESTS=1"),
]

REPO = "fastrtc/kokoro-onnx"
COMMIT = "8d07950c9b6c87ce6809e9bba7bd494336217c2a"
FILES = ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
SENTENCE = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def kokoro_82m() -> Path:
    from huggingface_hub import snapshot_download

    cache = os.environ.get("HF_HUB_CACHE") or str(REAL_HOME / ".cache/huggingface/hub")
    return Path(snapshot_download(REPO, revision=COMMIT, allow_patterns=FILES, cache_dir=cache))


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z ]", "", text.lower()).split()


def test_kokoro_speaks_and_whisper_hears_it(kokoro_82m: Path, whisper_small: Path) -> None:  # noqa: F811
    from haybale_audio.value import AudioBlock

    device, compute, notes = kokoro.resolve("cpu", "auto")
    tts = kokoro.load(kokoro_82m, device, compute, 0, name="Kokoro 82M", notes=notes)
    assert len(tts.voices) == 54

    samples = kokoro.speak(tts, SENTENCE, kokoro.voice(tts, "af_heart"))
    clip = AudioBlock(samples[None, :].copy(), tts.sample_rate)
    assert 1.5 < clip.duration < 6.0

    stt = load_whisper(whisper_small, "cpu", "int8", 0, name="Whisper small")
    result = transcribe(stt, model_input(clip, stt.sample_rate), language="en")
    assert _words(result.text) == _words(SENTENCE)


def test_kokoro_speaks_two_texts_at_once_as_it_would_one_after_the_other(kokoro_82m: Path) -> None:
    """ONNX Runtime runs one session from two threads; kokoro-onnx locks only espeak-ng."""
    import threading

    import numpy as np

    device, compute, notes = kokoro.resolve("cpu", "auto")
    tts = kokoro.load(kokoro_82m, device, compute, 2, name="Kokoro 82M", notes=notes, workers=2)
    voice = kokoro.voice(tts, "af_heart")
    texts = [SENTENCE, "Good evening. This is a different voice, speaking from the studio."]
    alone = [kokoro.speak(tts, text, voice) for text in texts]
    together: list[Any] = [None, None]

    def run(index: int) -> None:
        together[index] = kokoro.speak(tts, texts[index], voice)

    threads = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for one, both in zip(alone, together, strict=True):
        assert one.shape == both.shape and np.allclose(one, both, atol=1e-4)


def test_the_circle_runs_in_a_graph(
    library_system: object,
    tmp_path: Path,
    kokoro_82m: Path,
    whisper_small: Path,  # noqa: F811
) -> None:
    """Both catalog entries resolve to their snapshots: Speak's clip goes to Transcribe."""
    from haybale_models.consumer import models_state
    from haybale_models.keys import HF_VOLUME, MODELS_VOLUME
    from haybale_models.value.model_ref import ModelRef
    from haywire.core.volumes import VolumeStore, get_volume_store, set_volume_store

    from tests.conftest import add_node, run_from

    assert kokoro_82m.parents[2] == whisper_small.parents[2]  # one HF cache holds both
    previous = get_volume_store()
    store = VolumeStore(tmp_path / "volumes.toml")
    set_volume_store(store)
    try:
        store.ensure(MODELS_VOLUME, label="Models", description="", location=tmp_path / "models")
        store.ensure(HF_VOLUME, label="Hugging Face cache", description="", location=kokoro_82m.parents[2])
        (tmp_path / "models").mkdir()
        models_state().index.scan()

        from haywire.api.workbench import BaseGraph
        from haywire.core.graph.scheduler import SyncScheduler

        graph = BaseGraph(filestem="Circle", validation_scheduler=SyncScheduler())
        begin = add_node(graph, "haybale_core.nodes.events.begin_play_event", "BeginPlayEventNode")
        printer = add_node(graph, "haybale_core.nodes.print", "PrintNode")
        load_tts = add_node(graph, "haybale_speech.nodes.load_speech_model", "LoadSpeechModelNode")
        load_stt = add_node(graph, "haybale_speech.nodes.load_speech_model", "LoadSpeechModelNode")
        speak = add_node(graph, "haybale_speech.nodes.speak", "SpeakNode")
        hear = add_node(graph, "haybale_speech.nodes.transcribe", "TranscribeNode")
        graph.create_edge_wrapper(begin.node_id, "exec", printer.node_id, "exec")
        graph.create_edge_wrapper(load_tts.node_id, "loaded", speak.node_id, "model")
        graph.create_edge_wrapper(load_stt.node_id, "loaded", hear.node_id, "model")
        graph.create_edge_wrapper(speak.node_id, "audio", hear.node_id, "audio")
        graph.create_edge_wrapper(hear.node_id, "text", printer.node_id, "message")
        load_tts.ports["model"].set_value(ModelRef(catalog="haybale-speech:catalog:kokoro-82m"))
        load_stt.ports["model"].set_value(ModelRef(catalog="haybale-speech:catalog:whisper-small"))
        for load in (load_tts, load_stt):
            load.settings.device = "cpu"
        speak.ports["text"].set_value(SENTENCE)
        hear.ports["language"].set_value("en")

        run_from(graph, begin)

        text = hear.ports["text"].get_value()
        assert _words(text) == _words(SENTENCE), (text, load_tts.wrapper.warnings, load_stt.wrapper.warnings)
        assert models_state().pool.entries() == []
    finally:
        set_volume_store(previous)
