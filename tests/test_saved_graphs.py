"""Graphs saved with haybale-speech 0.0.2 still load, with the replaced nodes marked deprecated."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

#: examples/live_speech_to_text.haywire as released in 0.0.2: Speech Segmenter and Load STT Model.
LIVE_0_0_2 = Path(__file__).parent / "fixtures" / "live_speech_to_text-0.0.2.haywire"


def test_the_0_0_2_live_demo_loads_and_only_its_audio_to_transcribe_edge_breaks(graph: Any) -> None:
    assert graph.load_from_file(str(LIVE_0_0_2))
    graph.force_validation()
    nodes = {w.node.node_id: w.node for w in graph.list_node_wrappers()}
    kinds = sorted(type(n).__name__ for n in nodes.values())
    assert kinds == sorted(
        [
            "BeginPlayEventNode",
            "DeviceInNode",
            "AudioEventNode",
            "SpeechSegmenterNode",
            "AudioEventNode",
            "LoadSttModelNode",
            "TranscribeNode",
            "PrintNode",
        ]
    )
    deprecated = {type(n).__name__ for n in nodes.values() if type(n).class_identity.deprecation_warning}
    assert deprecated == {"SpeechSegmenterNode", "LoadSttModelNode"}

    edges = graph.list_edge_wrappers()
    assert len(edges) == 9  # every saved edge is kept
    broken = [
        (type(nodes[e.source_node_id]).__name__, e.outlet_port_id, type(nodes[e.sink_node_id]).__name__)
        for e in edges
        if not e.state.is_valid()
    ]
    # An Audio Event's block no longer connects to Transcribe, which takes an Audio Clip.
    assert broken == [("AudioEventNode", "audio", "TranscribeNode")]
