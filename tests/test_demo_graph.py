"""The live speech demo graph in examples/: it loads, and every edge is valid.

Rebuild the file after a port change with ``uv run --frozen python -m tests.test_demo_graph``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

DEMO = Path(__file__).parent.parent / "examples" / "live_speech_to_text.haywire"

#: (module, class, position) of each node, by name.
NODES = {
    "begin": ("haybale_core.nodes.events.begin_play_event", "BeginPlayEventNode", (0, 0)),
    "device_in": ("haybale_audio.nodes.device_in", "DeviceInNode", (260, 0)),
    "mic_event": ("haybale_audio.nodes.audio_event", "AudioEventNode", (0, 260)),
    "segmenter": ("haybale_speech.nodes.speech_segmenter", "SpeechSegmenterNode", (300, 260)),
    "clip_event": ("haybale_audio.nodes.audio_event", "AudioEventNode", (0, 640)),
    "load": ("haybale_speech.nodes.load_stt_model", "LoadSttModelNode", (0, 900)),
    "transcribe": ("haybale_speech.nodes.transcribe", "TranscribeNode", (300, 820)),
    "print": ("haybale_core.nodes.print", "PrintNode", (600, 640)),
}

#: (outlet node, outlet, inlet node, inlet).
EDGES = [
    ("begin", "exec", "device_in", "start"),
    ("mic_event", "listen_callback", "device_in", "callbacks"),
    ("mic_event", "block_ready", "segmenter", "add"),
    ("mic_event", "audio", "segmenter", "audio"),
    ("clip_event", "listen_callback", "segmenter", "callbacks"),
    ("clip_event", "block_ready", "print", "exec"),
    ("clip_event", "audio", "transcribe", "audio"),
    ("load", "loaded", "transcribe", "model"),
    ("transcribe", "text", "print", "message"),
]


def build_demo_graph(graph: Any) -> dict[str, Any]:
    """Place the demo's nodes and edges in *graph*; return the nodes by name."""
    nodes = {}
    for name, (module, cls_name, position) in NODES.items():
        cls = getattr(importlib.import_module(module), cls_name)
        wrapper = graph.create_node_wrapper(cls.class_identity.registry_key, position=position)
        assert wrapper is not None and isinstance(wrapper.node, cls), f"node creation failed for {name}"
        nodes[name] = wrapper.node
    for source, outlet, target, inlet in EDGES:
        graph.create_edge_wrapper(nodes[source].node_id, outlet, nodes[target].node_id, inlet)
    return nodes


def test_the_demo_graph_loads_with_every_edge_valid(graph: Any) -> None:
    assert graph.load_from_file(str(DEMO))
    graph.force_validation()
    kinds = sorted(type(w.node).__name__ for w in graph.list_node_wrappers())
    assert kinds == sorted(cls for _, cls, _ in NODES.values())
    edges = graph.list_edge_wrappers()
    assert len(edges) == len(EDGES)
    assert all(e.state.is_valid() for e in edges)


def test_the_demo_graph_is_what_build_demo_graph_makes(graph: Any) -> None:
    """The shipped file is current: the same wiring as the builder above."""
    assert graph.load_from_file(str(DEMO))
    by_id = {w.node.node_id: type(w.node).__name__ for w in graph.list_node_wrappers()}
    saved = sorted(
        (by_id[e.source_node_id], e.outlet_port_id, by_id[e.sink_node_id], e.inlet_port_id)
        for e in graph.list_edge_wrappers()
    )
    wanted = sorted((NODES[s][1], o, NODES[t][1], i) for s, o, t, i in EDGES)
    assert saved == wanted


if __name__ == "__main__":  # rebuild examples/live_speech_to_text.haywire
    from haywire.api.workbench import BaseGraph
    from haywire.core.di.config import set_global_injector, set_library_system
    from haywire.core.di.test_config import create_test_library_system
    from haywire.core.graph.scheduler import SyncScheduler

    root = Path(__file__).parent.parent
    service = create_test_library_system(
        workspace_root=str(root),
        library_paths=[str(root / "barn")],
        load_libraries=True,
        enable_file_watching=False,
    )
    set_library_system(service)
    set_global_injector(service.injector)
    demo = BaseGraph(filestem=DEMO.stem, validation_scheduler=SyncScheduler())
    build_demo_graph(demo)
    demo.force_validation()
    DEMO.parent.mkdir(exist_ok=True)
    assert demo.save_to_file(str(DEMO))
    print(f"wrote {DEMO}")
