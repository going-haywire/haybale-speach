"""The repo's haybale loads into a library system, with its components and catalog."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def test_speech_library_loads(library_system: Any) -> None:
    assert library_system.get_library_registry().is_library_enabled("haybale-speech")


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("haybale_speech.nodes.load_speech_model", "LoadSpeechModelNode"),
        ("haybale_speech.nodes.load_stt_model", "LoadSttModelNode"),
        ("haybale_speech.nodes.speak", "SpeakNode"),
        ("haybale_speech.nodes.pick_voice", "PickVoiceNode"),
        ("haybale_speech.nodes.transcribe", "TranscribeNode"),
        ("haybale_speech.nodes.collect_clip", "CollectClipNode"),
    ],
)
def test_nodes_place_in_a_graph(graph: Any, module: str, name: str) -> None:
    from tests.conftest import add_node

    add_node(graph, module, name)


def test_the_catalog_offers_whisper_small_pinned_and_commercial(library_system: Any) -> None:
    from haybale_models.consumer import models_state

    entry: Any = models_state().catalog.get("haybale-speech:catalog:whisper-small")
    assert entry is not None
    assert entry.purpose == ["stt"]
    assert entry.license == "MIT"
    assert entry.commercial is True
    (resource,) = entry.resources
    assert resource.backend == "faster-whisper"
    (url,) = resource.urls
    repo, _, commit = url.partition("@")
    assert repo == "hf://Systran/faster-whisper-small"
    assert len(commit) == 40  # a whole-repo address is pinned by its commit


def test_the_catalog_offers_kokoro_pinned_and_commercial(library_system: Any) -> None:
    from haybale_models.consumer import models_state

    entry: Any = models_state().catalog.get("haybale-speech:catalog:kokoro-82m")
    assert entry is not None
    assert entry.purpose == ["tts"]
    assert entry.license == "Apache-2.0"
    assert entry.commercial is True
    (resource,) = entry.resources
    assert resource.backend == "kokoro-onnx"
    assert resource.include == ("kokoro-v1.0.onnx", "voices-v1.0.bin")
    (url,) = resource.urls
    repo, _, commit = url.partition("@")
    assert repo == "hf://fastrtc/kokoro-onnx"
    assert len(commit) == 40


def test_load_stt_model_is_deprecated_for_load_speech_model(library_system: Any) -> None:
    from haybale_speech.nodes.load_stt_model import LoadSttModelNode

    assert "Load Speech Model" in LoadSttModelNode.class_identity.deprecation_warning


def test_the_speech_settings_have_a_panel(library_system: Any) -> None:
    registry = library_system.get_panel_registry()
    assert registry.get("haybale-speech:panel:SpeechSettingsPanel") is not None
