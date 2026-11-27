"""The demo graphs in examples/: each loads, every edge is valid, and its models are chosen.

Rebuild the files after a port change with ``uv run --frozen python -m tests.test_demo_graph``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).parent.parent / "examples"

WHISPER = "haybale-speech:catalog:whisper-small"
KOKORO = "haybale-speech:catalog:kokoro-82m"

#: The nodes both demos start with: the microphone cut into clips, one Clip Event per clip.
LISTEN_NODES = {
    "begin": ("haybale_core.nodes.events.begin_play_event", "BeginPlayEventNode", (0, 0)),
    "device_in": ("haybale_audio.nodes.device_in", "DeviceInNode", (260, 0)),
    "mic_event": ("haybale_audio.nodes.audio_event", "AudioEventNode", (0, 260)),
    "segmenter": ("haybale_audio.nodes.clip_segmenter", "ClipSegmenterNode", (300, 260)),
    "clip_event": ("haybale_audio.nodes.clip_event", "ClipEventNode", (0, 640)),
    "stt": ("haybale_speech.nodes.load_speech_model", "LoadSpeechModelNode", (0, 900)),
    "transcribe": ("haybale_speech.nodes.transcribe", "TranscribeNode", (300, 820)),
    "print": ("haybale_core.nodes.print", "PrintNode", (600, 640)),
}

LISTEN_EDGES = [
    ("begin", "exec", "device_in", "start"),
    ("mic_event", "listen_callback", "device_in", "callbacks"),
    ("mic_event", "block_ready", "segmenter", "add"),
    ("mic_event", "audio", "segmenter", "audio"),
    ("clip_event", "listen_callback", "segmenter", "callbacks"),
    ("clip_event", "clip_ready", "print", "exec"),
    ("clip_event", "clip", "transcribe", "audio"),
    ("stt", "loaded", "transcribe", "model"),
    ("transcribe", "text", "print", "message"),
]


@dataclass(frozen=True)
class Demo:
    """One demo file: its nodes (name → module, class, position), edges and models (name → catalog)."""

    file: Path
    nodes: dict[str, tuple[str, str, tuple[int, int]]]
    edges: list[tuple[str, str, str, str]]
    models: dict[str, str]


LIVE = Demo(EXAMPLES / "live_speech_to_text.haywire", LISTEN_NODES, LISTEN_EDGES, {"stt": WHISPER})

#: Each clip heard is printed, spoken back by Kokoro and played on the output device.
ROUND_TRIP = Demo(
    EXAMPLES / "speech_round_trip.haywire",
    {
        **LISTEN_NODES,
        "tts": ("haybale_speech.nodes.load_speech_model", "LoadSpeechModelNode", (300, 1100)),
        "speak": ("haybale_speech.nodes.speak", "SpeakNode", (600, 900)),
        "player": ("haybale_audio.nodes.clip_player", "ClipPlayerNode", (900, 640)),
        "out_event": ("haybale_audio.nodes.audio_event", "AudioEventNode", (900, 260)),
        "device_out": ("haybale_audio.nodes.device_out", "DeviceOutNode", (1200, 260)),
    },
    [
        *LISTEN_EDGES,
        ("print", "done", "player", "start"),
        ("transcribe", "text", "speak", "text"),
        ("tts", "loaded", "speak", "model"),
        ("speak", "audio", "player", "clip"),
        ("out_event", "listen_callback", "player", "callbacks"),
        ("out_event", "block_ready", "device_out", "write"),
        ("out_event", "audio", "device_out", "audio"),
    ],
    {"stt": WHISPER, "tts": KOKORO},
)

DEMOS = [LIVE, ROUND_TRIP]


def build_demo_graph(graph: Any, demo: Demo) -> dict[str, Any]:
    """Place *demo*'s nodes and edges in *graph*; return the nodes by name."""
    nodes = {}
    for name, (module, cls_name, position) in demo.nodes.items():
        cls = getattr(importlib.import_module(module), cls_name)
        wrapper = graph.create_node_wrapper(cls.class_identity.registry_key, position=position)
        assert wrapper is not None and isinstance(wrapper.node, cls), f"node creation failed for {name}"
        nodes[name] = wrapper.node
    for source, outlet, target, inlet in demo.edges:
        graph.create_edge_wrapper(nodes[source].node_id, outlet, nodes[target].node_id, inlet)
    from haybale_models.value.model_ref import ModelRef

    for name, catalog in demo.models.items():
        nodes[name].ports["model"].set_value(ModelRef(catalog=catalog))
    return nodes


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.file.stem)
def test_the_demo_graph_loads_with_every_edge_valid(graph: Any, demo: Demo) -> None:
    assert graph.load_from_file(str(demo.file))
    graph.force_validation()
    kinds = sorted(type(w.node).__name__ for w in graph.list_node_wrappers())
    assert kinds == sorted(cls for _, cls, _ in demo.nodes.values())
    edges = graph.list_edge_wrappers()
    assert len(edges) == len(demo.edges)
    assert all(e.state.is_valid() for e in edges)


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.file.stem)
def test_the_demo_graph_is_what_build_demo_graph_makes(graph: Any, demo: Demo) -> None:
    """The shipped file is current: the same wiring as the builder above."""
    assert graph.load_from_file(str(demo.file))
    by_id = {w.node.node_id: type(w.node).__name__ for w in graph.list_node_wrappers()}
    saved = sorted(
        (by_id[e.source_node_id], e.outlet_port_id, by_id[e.sink_node_id], e.inlet_port_id)
        for e in graph.list_edge_wrappers()
    )
    wanted = sorted((demo.nodes[s][1], o, demo.nodes[t][1], i) for s, o, t, i in demo.edges)
    assert saved == wanted


@pytest.mark.parametrize("demo", DEMOS, ids=lambda d: d.file.stem)
def test_the_demo_graph_picks_its_models(graph: Any, demo: Demo) -> None:
    """Each Load Speech Model keeps its kind and its catalog entry: no ``ModelMissing`` on start."""
    from haybale_speech.types.stt_model import STT_MODEL
    from haybale_speech.types.tts_model import TTS_MODEL

    assert graph.load_from_file(str(demo.file))
    loaders = [w.node for w in graph.list_node_wrappers() if type(w.node).__name__ == "LoadSpeechModelNode"]
    found = {}
    for load in loaders:
        kind = load.ports["loaded"].type_cls
        assert kind in (STT_MODEL, TTS_MODEL)
        purpose = "stt" if kind is STT_MODEL else "tts"
        assert load.ports["model"].widget_config["properties"]["filter"] == {"purpose": [purpose]}
        found[purpose] = load.ports["model"].get_value().catalog
    wanted = {"stt" if c == WHISPER else "tts": c for c in demo.models.values()}
    assert found == wanted


if __name__ == "__main__":  # rebuild the demo files in examples/
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
    EXAMPLES.mkdir(exist_ok=True)
    for demo in DEMOS:
        built = BaseGraph(filestem=demo.file.stem, validation_scheduler=SyncScheduler())
        build_demo_graph(built, demo)
        built.force_validation()
        assert built.save_to_file(str(demo.file))
        print(f"wrote {demo.file}")
