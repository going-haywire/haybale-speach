"""The real Whisper small model transcribes the speech fixture.

Needs the model in the Hugging Face cache (about 490 MB), so it runs only with
``SPEECH_MODEL_TESTS=1``; with that set, a missing snapshot is downloaded first.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import soundfile

from haybale_speech.runtime.audio import model_input
from haybale_speech.runtime.whisper import load_whisper, transcribe
from tests.conftest import REAL_HOME

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get("SPEECH_MODEL_TESTS"), reason="set SPEECH_MODEL_TESTS=1"),
]

REPO = "Systran/faster-whisper-small"
COMMIT = "536b0662742c02347bc0e980a01041f333bce120"
FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]


@pytest.fixture(scope="module")
def whisper_small() -> Path:
    from huggingface_hub import snapshot_download

    cache = os.environ.get("HF_HUB_CACHE") or str(REAL_HOME / ".cache/huggingface/hub")
    return Path(snapshot_download(REPO, revision=COMMIT, allow_patterns=FILES, cache_dir=cache))


def test_whisper_small_transcribes_the_fixture(whisper_small: Path, speech_wav: Path) -> None:
    from haybale_audio.value import AudioBlock

    data, rate = soundfile.read(speech_wav, dtype="float32", always_2d=True)
    clip = AudioBlock(data.T.copy(), rate)
    model = load_whisper(whisper_small, "cpu", "int8", 0, name="Whisper small")

    result = transcribe(model, model_input(clip, model.sample_rate))

    words = re.sub(r"[^a-z ]", "", result.text.lower()).split()
    assert words == "the quick brown fox jumps over the lazy dog".split()
    assert result.language == "en"
    assert result.segments and result.segments[-1].end <= result.duration + 0.5


def test_the_catalog_entry_runs_in_a_graph(
    library_system: object, tmp_path: Path, whisper_small: Path, speech_wav: Path
) -> None:
    """Load STT Model resolves the catalog entry to the snapshot in the HF cache; Transcribe runs it."""
    from haybale_audio.value import AudioBlock
    from haybale_models.consumer import models_state
    from haybale_models.keys import HF_VOLUME, MODELS_VOLUME
    from haybale_models.value.model_ref import ModelRef
    from haywire.core.volumes import VolumeStore, get_volume_store, set_volume_store

    from tests.conftest import add_node, run_from

    previous = get_volume_store()
    store = VolumeStore(tmp_path / "volumes.toml")
    set_volume_store(store)
    try:
        store.ensure(MODELS_VOLUME, label="Models", description="", location=tmp_path / "models")
        store.ensure(
            HF_VOLUME, label="Hugging Face cache", description="", location=whisper_small.parents[2]
        )
        (tmp_path / "models").mkdir()
        models_state().index.scan()

        from haywire.api.workbench import BaseGraph
        from haywire.core.graph.scheduler import SyncScheduler

        graph = BaseGraph(filestem="Whisper", validation_scheduler=SyncScheduler())
        begin = add_node(graph, "haybale_core.nodes.events.begin_play_event", "BeginPlayEventNode")
        printer = add_node(graph, "haybale_core.nodes.print", "PrintNode")
        load = add_node(graph, "haybale_speech.nodes.load_stt_model", "LoadSttModelNode")
        node = add_node(graph, "haybale_speech.nodes.transcribe", "TranscribeNode")
        graph.create_edge_wrapper(begin.node_id, "exec", printer.node_id, "exec")
        graph.create_edge_wrapper(load.node_id, "loaded", node.node_id, "model")
        graph.create_edge_wrapper(node.node_id, "text", printer.node_id, "message")
        load.ports["model"].set_value(ModelRef(catalog="haybale-speech:catalog:whisper-small"))
        load.settings.device = "cpu"
        data, rate = soundfile.read(speech_wav, dtype="float32", always_2d=True)
        node.ports["audio"].set_value(AudioBlock(data.T.copy(), rate))

        run_from(graph, begin)

        text = node.ports["text"].get_value()
        assert "quick brown fox" in text.lower(), (text, load.wrapper.warnings)
        assert models_state().pool.entries() == []
    finally:
        set_volume_store(previous)
