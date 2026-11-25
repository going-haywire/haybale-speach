"""Speech Segmenter — cuts a live audio flow into speech clips at the pauses."""

from __future__ import annotations

import time

from haybale_audio.emitter import STATUS_INTERVAL, deliver_to_subscribers, set_status
from haybale_audio.value import AudioBlock
from haywire.api import FLOAT, INT, STRING, BaseNode, ExecutionContext, LabelWidget, NodeType, node

from haybale_speech.runtime.segmenter import Segmenter, speech_clip


@node(
    label="Speech Segmenter",
    description="Cuts a live audio flow into speech clips at the pauses, for Transcribe",
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
)
class SpeechSegmenterNode(BaseNode):
    """Listens to an audio flow and sends each stretch of speech as a clip: mono, 16 kHz.

    A block is loud when its level (RMS) is above `Threshold`. The first loud block opens a
    segment, together with the `Pre-roll` before it, so a quiet first sound is kept. A pause
    of `Hang-over` ends the segment and the clip goes to every Audio Event listening on
    `Subscribers`; the pause itself is left out. Speech shorter than `Min Seconds` (a click, a
    cough) is discarded; a segment that reaches `Max Seconds` is sent at once and the next loud
    block opens a new one. The status shows the level, which helps to set the threshold.

    The clip arrives in a flow of its own, so transcribing it never holds up the audio flow,
    however long the model takes:

    ```text
    Device In → Audio Event → Speech Segmenter (Add)
    Speech Segmenter (Subscribers) ← Audio Event → Print, with Transcribe's Text as the message
    ```

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

    def on_shutdown(self, context: ExecutionContext) -> None:
        self.hb_reset()

    def on_teardown(self) -> None:
        self.hb_reset()

    def hb_reset(self) -> None:
        # post_init() may never have run (init() failed during a reset or reload).
        segmenter = getattr(self, "hb_segmenter", None)
        if segmenter is not None:
            segmenter.reset()

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
        segmenter.threshold = float(threshold)
        segmenter.hangover = max(0, int(hangover_ms)) / 1000
        segmenter.preroll = max(0, int(preroll_ms)) / 1000
        segmenter.min_length = float(min_seconds)
        segmenter.max_length = max(float(max_seconds), audio.duration)
        segment = segmenter.feed(audio)
        if segment is not None:
            subscribers = self.value("callbacks")
            deliver_to_subscribers(
                context, subscribers, speech_clip(segment.audio), segmenter.sent, segment.start
            )
            self.hb_report(
                force=True, note=f"sent {segment.duration:.1f} s" if subscribers else "no subscriber"
            )
        else:
            self.hb_report()
        return "next"

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
