"""The deprecated Speech Segmenter node, on synthetic blocks: a tone for speech, silence for pauses.

The segmenting itself is haybale-audio's (``haybale_audio.dsp.segmenter``), tested there.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from haybale_audio.value import AudioBlock

from haybale_speech.nodes.speech_segmenter import SPEECH_RATE, speech_clip
from tests.conftest import add_node

RATE = 48000
BLOCK = 480  # 10 ms at 48 kHz, so lengths read as whole blocks
SEGMENTER = ("haybale_speech.nodes.speech_segmenter", "SpeechSegmenterNode")


def _tone(blocks: int = 1, channels: int = 1, amplitude: float = 0.1) -> list[AudioBlock]:
    out = []
    for n in range(blocks):
        t = (np.arange(BLOCK) + n * BLOCK) / RATE
        wave = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out.append(AudioBlock(np.tile(wave, (channels, 1)), RATE))
    return out


def _silence(blocks: int = 1, channels: int = 1, level: float = 0.0) -> list[AudioBlock]:
    return [AudioBlock(np.full((channels, BLOCK), level, np.float32), RATE) for _ in range(blocks)]


@pytest.mark.unit
def test_the_clip_is_mono_at_16k_and_keeps_its_length() -> None:
    stereo = AudioBlock(np.concatenate([b.data for b in _tone(100, channels=2)], axis=1), RATE)
    clip = speech_clip(stereo)
    assert (clip.channels, clip.sample_rate) == (1, SPEECH_RATE)
    assert abs(clip.frames - SPEECH_RATE) <= 2
    assert clip.data.dtype == np.float32


# --- the node ----------------------------------------------------------------


def _segmenter_node(graph: Any, subscribed: bool = True) -> Any:
    node = add_node(graph, *SEGMENTER)
    if subscribed:
        cb = importlib.import_module("haybale_audio.types.callback").AUDIO_CALLBACK
        node.value = lambda port_id: {"edge": cb(name="listener")} if port_id == "callbacks" else None
    return node


def _context() -> MagicMock:
    ctx = MagicMock()
    ctx.control_pin = "add"
    return ctx


def _run(node: Any, ctx: Any, blocks: list[AudioBlock], **settings: Any) -> None:
    values = {
        "threshold": 0.01,
        "hangover_ms": 100,
        "preroll_ms": 0,
        "min_seconds": 0.1,
        "max_seconds": 30.0,
    }
    values.update(settings)
    for block in blocks:
        assert node.worker(ctx, audio=block, **values) == "next"


@pytest.mark.integration
def test_node_sends_each_segment_as_a_16k_mono_clip_to_its_subscribers(graph: Any) -> None:
    node = _segmenter_node(graph)
    ctx = _context()
    _run(node, ctx, _silence(10, channels=2) + _tone(50, channels=2) + _silence(10, channels=2))
    _run(node, ctx, _tone(30, channels=2) + _silence(10, channels=2))

    calls = ctx.emit_callback.call_args_list
    assert len(calls) == 2
    first = calls[0].kwargs
    assert first["event_name"] == ["listener"]
    clip = first["payload"]["audio"]
    assert (clip.channels, clip.sample_rate) == (1, SPEECH_RATE)
    assert abs(clip.frames - SPEECH_RATE // 2) <= 2  # 50 blocks of 10 ms
    assert first["payload"]["timestamp"] == pytest.approx(0.1)
    assert first["payload"]["block_number"] == 1
    assert calls[1].kwargs["payload"]["block_number"] == 2
    assert node.ports["status"].get_value() == "Listening, level 0.000, sent 2, discarded 0 (sent 0.3 s)"


@pytest.mark.integration
def test_node_settings_apply_on_the_next_block(graph: Any) -> None:
    node = _segmenter_node(graph)
    ctx = _context()
    _run(node, ctx, _tone(5) + _silence(10), threshold=0.5)  # the tone is below this threshold
    assert ctx.emit_callback.call_count == 0
    _run(node, ctx, _tone(15) + _silence(10), threshold=0.01)
    assert ctx.emit_callback.call_count == 1


@pytest.mark.integration
def test_node_without_subscriber_segments_but_sends_nothing(graph: Any) -> None:
    node = add_node(graph, *SEGMENTER)  # its Subscribers inlet is not wired
    ctx = _context()
    _run(node, ctx, _tone(20) + _silence(10))
    assert ctx.emit_callback.call_count == 0
    assert node.ports["status"].get_value().endswith("sent 1, discarded 0 (no subscriber)")


@pytest.mark.integration
def test_node_drops_an_open_segment_on_shutdown(graph: Any) -> None:
    node = _segmenter_node(graph)
    ctx = _context()
    _run(node, ctx, _tone(20))
    node.on_shutdown(ctx)
    _run(node, ctx, _silence(10))
    assert ctx.emit_callback.call_count == 0


@pytest.mark.integration
def test_node_without_audio_is_an_error(graph: Any) -> None:
    node = _segmenter_node(graph)
    with pytest.raises(ValueError, match="audio block"):
        node.worker(_context(), audio=None)
