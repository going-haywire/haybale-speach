"""Speech Segmenter — the clip cutter of 0.0.2, kept so saved graphs load. Use Clip Segmenter."""

from __future__ import annotations

import time

from haybale_audio.dsp.segmenter import Segmenter
from haybale_audio.emitter import STATUS_INTERVAL, deliver_to_subscribers, set_status
from haybale_audio.value import AudioBlock
from haywire.api import FLOAT, INT, STRING, BaseNode, ExecutionContext, LabelWidget, NodeType, node

from haybale_speech.runtime.audio import model_input

#: The sample rate of the clips the segmenter sends: the rate speech models take.
SPEECH_RATE = 16000


def speech_clip(segment: AudioBlock) -> AudioBlock:
    """Return *segment* mixed to mono and resampled to :data:`SPEECH_RATE`, as one block."""
    return AudioBlock(model_input(segment, SPEECH_RATE)[None, :], SPEECH_RATE)


@node(
    label="Speech Segmenter",
    description="Cuts a live audio flow into speech clips for Audio Events; replaced by Clip Segmenter",
    menu="speech",
    search_tags=[
        "audio",
        "speech",
        "vad",
        "voice",
        "segment",
        "split",
        "pause",
        "silence",
        "live",
        "microphone",
    ],
    node_type=NodeType.CONTROL,
    deprecation_warning=(
        "Speech Segmenter is replaced by Clip Segmenter (haybale-audio, menu audio/clip): its clips go "
        "to Clip Events, whose Clip connects to Transcribe"
    ),
)
class SpeechSegmenterNode(BaseNode):
    """Sends each stretch of speech in an audio flow to Audio Events, as a mono 16 kHz block.
    Replaced by Clip Segmenter.

    Graphs saved with it keep loading, but an Audio Event's `Audio` no longer connects to
    Transcribe, which takes an Audio Clip: rebuild such a graph with haybale-audio's Clip
    Segmenter and a Clip Event, whose `Clip` goes to Transcribe.

    A block is loud when its level (RMS) is above `Threshold`. The first loud block opens a
    segment, together with the `Pre-roll` before it, so a quiet first sound is kept. A pause
    of `Hang-over` ends the segment and the clip goes to every Audio Event listening on
    `Subscribers`; the pause itself is left out. Speech shorter than `Min Seconds` (a click, a
    cough) is discarded; a segment that reaches `Max Seconds` is sent at once and the next loud
    block opens a new one. The status shows the level, which helps to set the threshold.

    The listening Audio Event's `Audio` is the clip, its `Timestamp` the start of the speech in
    seconds since the segmenter's first block, and its `Block Number` the count of clips sent.

    Inputs:
        add: Feed the block on `audio` (wire it to an Audio Event's Block Ready)
        audio: The block
        callbacks: Pooled AUDIO_CALLBACK subscriptions from the Audio Event nodes that get the clips
        threshold: Level (RMS) above which a block counts as speech
        hangover_ms: Pause that ends a segment
        preroll_ms: Audio kept from before the first loud block
        min_seconds: Shortest speech sent; shorter is discarded
        max_seconds: Longest segment; a longer one is cut there

    Outputs:
        next: Control flow after the block is taken
    """

    def init(self) -> None:
        from haybale_audio.types.audio import AUDIO
        from haybale_audio.types.audio_exec import AUDIO_EXEC
        from haybale_audio.types.callback import AUDIO_CALLBACK
        from haybale_core.types import POOLED

        self.add(AUDIO_EXEC.as_inlet("add", label="Add"))
        self.add(AUDIO.as_inlet("audio", label="Audio", description="Block to listen to"))
        self.add(
            POOLED[AUDIO_CALLBACK].as_inlet(
                "callbacks",
                label="Subscribers",
                description="Connect to the Audio Event nodes that get the clips.",
            )
        )
        self.add(
            FLOAT.as_config(
                "threshold",
                default=0.01,
                label="Threshold",
                description="Level (RMS) above which a block counts as speech",
            )
        )
        self.add(
            INT.as_config(
                "hangover_ms", default=500, label="Hang-over (ms)", description="Pause that ends a segment"
            )
        )
        self.add(
            INT.as_config(
                "preroll_ms",
                default=200,
                label="Pre-roll (ms)",
                description="Audio kept from before the first loud block",
            )
        )
        self.add(
            FLOAT.as_config(
                "min_seconds",
                default=0.3,
                label="Min Seconds",
                description="Shortest speech sent; shorter is discarded",
            )
        )
        self.add(
            FLOAT.as_config(
                "max_seconds",
                default=30.0,
                label="Max Seconds",
                description="Longest segment; a longer one is cut there",
            )
        )
        self.add(STRING.as_config("status", default="Idle", label="Status", widget=LabelWidget.config()))
        self.add(AUDIO_EXEC.as_outlet("next", label="Next"))

    def post_init(self) -> None:
        self.hb_segmenter = Segmenter()
        self.hb_status_at = 0.0
        self.hb_channels = 0  # channel count of the stream so far
        self.hb_rate = 0  # sample rate of the stream
        self.hb_seconds = 0.0  # stream time fed so far
        self.hb_offset = 0.0  # stream time at the segmenter's last restart

    def on_shutdown(self, context: ExecutionContext) -> None:
        self.hb_reset()

    def on_teardown(self) -> None:
        self.hb_reset()

    def hb_reset(self) -> None:
        # post_init() may never have run (init() failed during a reset or reload).
        segmenter = getattr(self, "hb_segmenter", None)
        if segmenter is not None:
            segmenter.reset()
            self.hb_channels, self.hb_rate, self.hb_seconds, self.hb_offset = 0, 0, 0.0, 0.0

    def worker(
        self,
        context: ExecutionContext,
        audio: AudioBlock | None = None,
        threshold: float = 0.01,
        hangover_ms: int = 500,
        preroll_ms: int = 200,
        min_seconds: float = 0.3,
        max_seconds: float = 30.0,
    ) -> str | None:
        if audio is None:
            raise ValueError("add needs an audio block")
        segmenter = self.hb_segmenter
        self.hb_take_format(audio)
        segmenter.threshold = float(threshold)
        segmenter.hangover = max(0, int(hangover_ms)) / 1000
        segmenter.preroll = max(0, int(preroll_ms)) / 1000
        segmenter.min_length = float(min_seconds)
        segmenter.max_length = max(float(max_seconds), audio.duration)
        segment = segmenter.feed(audio)  # raises on another rate, or new channels inside a clip
        self.hb_channels, self.hb_rate = audio.channels, audio.sample_rate
        self.hb_seconds += audio.duration
        if segment is not None:
            subscribers = self.value("callbacks")
            deliver_to_subscribers(
                context,
                subscribers,
                speech_clip(segment.audio),
                segmenter.sent,
                self.hb_offset + segment.start,
            )
            self.hb_report(
                force=True, note=f"sent {segment.duration:.1f} s" if subscribers else "no subscriber"
            )
        else:
            self.hb_report()
        return "next"

    def hb_take_format(self, audio: AudioBlock) -> None:
        """Let a new channel count start between two clips, as in 0.0.2.

        haybale-audio's segmenter refuses a channel change anywhere in the stream; 0.0.2's
        took a new one with each clip. So between clips the segmenter starts over, keeping its
        counts and the stream time. Inside a clip a change is still an error, and so is
        another sample rate, as in 0.0.2.
        """
        segmenter = self.hb_segmenter
        new_channels = bool(self.hb_channels) and audio.channels != self.hb_channels
        same_rate = audio.sample_rate == self.hb_rate
        if new_channels and same_rate and not segmenter.speaking:
            sent, discarded = segmenter.sent, segmenter.discarded
            segmenter.reset()
            segmenter.sent, segmenter.discarded = sent, discarded
            self.hb_offset = self.hb_seconds

    def hb_report(self, force: bool = False, note: str = "") -> None:
        now = time.monotonic()
        if not force and now - self.hb_status_at < STATUS_INTERVAL:
            return
        self.hb_status_at = now
        segmenter = self.hb_segmenter
        state = f"Speaking {segmenter.seconds:.1f} s" if segmenter.speaking else "Listening"
        text = (
            f"{state}, level {segmenter.level:.3f}, sent {segmenter.sent}, discarded {segmenter.discarded}"
        )
        set_status(self, f"{text} ({note})" if note else text)
