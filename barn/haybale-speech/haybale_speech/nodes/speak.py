"""Speak — turn text into an audio clip with a loaded text-to-speech model."""

from __future__ import annotations

import numpy as np
from haybale_audio.value import AudioBlock
from haywire.api import (
    FLOAT,
    STRING,
    BaseNode,
    ExecutionContext,
    MenuWidget,
    NodeSettings,
    NodeType,
    PortType,
    ProgressSettings,
    Promotable,
    SliderWidget,
    bag,
    node,
    progress_field,
    setting,
)

from haybale_speech.runtime.kokoro import SPEED_RANGE, speak, voice as model_voice
from haybale_speech.runtime.whisper import Cancelled
from haybale_speech.value import TtsModel, Voice

#: Suggested voices (Kokoro's); any voice the model has can be typed in.
VOICES = ("af_heart", "af_bella", "am_michael", "bf_emma", "bm_george", "ff_siwis", "if_sara", "ef_dora")


class VoiceSettings(NodeSettings):
    """Which of the model's voices speaks. A port by default; promotable to an inlet or an outlet."""

    voice_name = setting[STRING](
        VOICES[0],
        label="Voice Name",
        category="Voice",
        description="The model's voice, such as af_heart; the first letter of a Kokoro voice is its language",
        widget=MenuWidget.config(properties={"options": list(VOICES), "new_value_mode": "add-unique"}),
        promotable=Promotable.ALL,
        promote_default=PortType.CONFIG,
    )


def run_again_on_change(node: BaseNode, settings: NodeSettings) -> None:
    """Make the data node *node* run again whenever a field of *settings* changes.

    A settings write marks no port dirty, so without this a data node whose inlets do not
    change keeps its last outlet while the graph runs.
    """
    settings._subscribe(lambda name, value, old: node.run_again())


class SpeakProgress(ProgressSettings):
    """How much of the text is spoken."""

    main = progress_field("Text")


@node(
    label="Speak",
    description="Turns text into an audio clip with a loaded text-to-speech model",
    menu="speech",
    search_tags=[
        "speech",
        "tts",
        "speak",
        "say",
        "voice",
        "text to speech",
        "synthesize",
        "kokoro",
        "read aloud",
    ],
    node_type=NodeType.DATA,
)
class SpeakNode(BaseNode):
    """Turns text into an audio clip with a loaded text-to-speech model.

    Connect `Model` to Load Speech Model and `Text` to the text to speak, such as Transcribe's
    `Text`. The clip is mono, at the model's sample rate (24 kHz for Kokoro).

    - `Voice Name`: the voice by name, such as `af_heart` (American English) or `bf_emma` (British
      English); the first letter of a Kokoro voice is its language. A `Voice` wire, from Pick
      Voice, takes its place. It is a setting, shown on the card: promote it to an inlet to choose
      the voice from another node while the graph runs, or to an outlet to pass it on.
    - `Speed`: 0.5 to 2.0; 1.0 is the voice's own pace.

    `Clip` is the spoken clip and `Seconds` its length; Clip Player plays it. The text is spoken
    one sentence at a time; stopping the graph stops after the current sentence.
    """

    settings = bag(VoiceSettings)
    progress = bag(SpeakProgress)

    def post_init(self) -> None:
        run_again_on_change(self, self.settings)

    def init(self) -> None:
        from haybale_audio.types.audio_clip import AUDIO_CLIP

        from haybale_speech.types.tts_model import TTS_MODEL
        from haybale_speech.types.voice import VOICE

        self.add(STRING.as_inlet("text", label="Text", description="The text to speak"))
        self.add(TTS_MODEL.as_inlet("model", label="Model", description="A loaded text-to-speech model"))
        self.add(
            VOICE.as_inlet(
                "voice", label="Voice", description="A voice from Pick Voice; replaces Voice Name"
            )
        )
        low, high = SPEED_RANGE
        self.add(
            FLOAT.as_config(
                "speed",
                default=1.0,
                label="Speed",
                widget=SliderWidget.config(properties={"min": low, "max": high, "step": 0.05}),
                description="0.5 to 2.0; 1.0 is the voice's own pace",
            )
        )
        self.add(AUDIO_CLIP.as_outlet("audio", label="Clip", description="The spoken clip, mono"))
        self.add(FLOAT.as_outlet("seconds", label="Seconds", description="Length of the clip"))

    def worker(
        self,
        context: ExecutionContext,
        text: str,
        model: TtsModel | None,
        voice: Voice | None,
        speed: float,
    ) -> None:
        if model is None:
            raise ValueError("no model — connect Load Speech Model")
        chosen = voice if voice is not None else model_voice(model, str(self.settings.voice_name))
        with self.progress.busy("main", "speaking") as span:
            try:
                samples = speak(
                    model,
                    text or "",
                    chosen,
                    speed=float(speed),
                    cancel=context.cancel,
                    on_progress=lambda share, count: span.step(share, f"{count} sentences"),
                )
            except Cancelled:
                return None
        clip = AudioBlock(np.ascontiguousarray(samples[np.newaxis, :]), model.sample_rate)
        self.out("audio", clip)
        self.out("seconds", clip.duration)
        return None
