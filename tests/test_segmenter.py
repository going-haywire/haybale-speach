"""The segmenter and the Speech Segmenter node, on synthetic blocks: a tone for speech, silence for pauses."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from haybale_audio.value import AudioBlock

from haybale_speech.runtime.segmenter import SPEECH_RATE, Segment, Segmenter, rms, speech_clip
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


def _feed(segmenter: Segmenter, blocks: list[AudioBlock]) -> list[Segment]:
    return [s for s in (segmenter.feed(b) for b in blocks) if s is not None]


@pytest.mark.unit
def test_rms_of_a_tone_and_of_silence() -> None:
    assert rms(_tone(amplitude=0.1)[0]) == pytest.approx(0.1 / np.sqrt(2), rel=0.01)
    assert rms(_silence()[0]) == 0.0
    assert rms(AudioBlock(np.zeros((1, 0), np.float32), RATE)) == 0.0


@pytest.mark.unit
def test_a_pause_of_the_hangover_ends_the_segment_without_the_pause() -> None:
    seg = Segmenter(threshold=0.01, hangover=0.1, preroll=0.0, min_length=0.0)
    out = _feed(seg, _silence(5) + _tone(30) + _silence(9))
    assert out == []
    assert seg.speaking
    (segment,) = _feed(seg, _silence(1))  # the 10th silent block: 100 ms
    assert segment.audio.frames == 30 * BLOCK  # the tone only, the pause left out
    assert segment.start == pytest.approx(5 * BLOCK / RATE)
    assert not seg.speaking
    assert seg.sent == 1


@pytest.mark.unit
def test_a_pause_shorter_than_the_hangover_stays_in_the_segment() -> None:
    seg = Segmenter(threshold=0.01, hangover=0.1, preroll=0.0, min_length=0.0)
    (segment,) = _feed(seg, _tone(20) + _silence(5) + _tone(20) + _silence(10))
    assert segment.audio.frames == 45 * BLOCK
    assert np.allclose(segment.audio.data[:, 20 * BLOCK : 25 * BLOCK], 0.0)


@pytest.mark.unit
def test_the_preroll_keeps_the_blocks_before_the_first_loud_one() -> None:
    seg = Segmenter(threshold=0.01, hangover=0.1, preroll=0.05, min_length=0.0)
    quiet = _silence(20, level=0.005)  # below the threshold, like a soft onset
    (segment,) = _feed(seg, quiet + _tone(30) + _silence(10))
    assert segment.audio.frames == (5 + 30) * BLOCK  # 50 ms of pre-roll
    assert np.allclose(segment.audio.data[:, : 5 * BLOCK], 0.005)
    assert segment.start == pytest.approx(15 * BLOCK / RATE)


@pytest.mark.unit
def test_speech_shorter_than_the_minimum_is_discarded() -> None:
    seg = Segmenter(threshold=0.01, hangover=0.1, preroll=0.05, min_length=0.2)
    # 15 tone blocks are 150 ms: with the 50 ms pre-roll the clip would reach 200 ms, but the
    # minimum counts the speech alone.
    assert _feed(seg, _silence(10) + _tone(15) + _silence(10)) == []
    assert seg.discarded == 1
    (segment,) = _feed(seg, _tone(20) + _silence(10))
    assert segment.audio.frames == (5 + 20) * BLOCK
    assert seg.sent == 1


@pytest.mark.unit
def test_a_segment_reaching_the_maximum_is_sent_and_the_speech_goes_on_in_a_new_one() -> None:
    seg = Segmenter(threshold=0.01, hangover=0.1, preroll=0.0, min_length=0.0, max_length=0.5)
    first, second = _feed(seg, _tone(80) + _silence(10))
    assert first.audio.frames == 50 * BLOCK
    assert second.audio.frames == 30 * BLOCK
    assert second.start == pytest.approx(0.5)


@pytest.mark.unit
def test_a_block_of_another_rate_is_an_error_and_reset_starts_over() -> None:
    seg = Segmenter()
    seg.feed(_tone()[0])
    with pytest.raises(ValueError, match="16000 Hz"):
        seg.feed(AudioBlock(np.zeros((1, 160), np.float32), 16000))
    seg.reset()
    assert not seg.speaking and seg.sent == 0
    seg.feed(AudioBlock(np.zeros((1, 160), np.float32), 16000))  # a new stream takes a new rate


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
    assert first["event_name"] == "listener"
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
