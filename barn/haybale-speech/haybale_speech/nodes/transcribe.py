"""Transcribe — turn the speech in an audio clip into text."""

from __future__ import annotations

from haybale_audio.value import AudioBlock
from haywire.api import (
    STRING,
    BaseNode,
    ExecutionContext,
    MenuWidget,
    NodeType,
    ProgressSettings,
    bag,
    node,
    progress_field,
)

from haybale_speech.runtime.audio import model_input
from haybale_speech.runtime.whisper import Cancelled, transcribe
from haybale_speech.value import SttModel

#: Suggested languages; any language code the model knows can be typed in.
LANGUAGES = ("auto", "en", "de", "fr", "it", "es", "pt", "nl", "ja", "zh")


class TranscribeProgress(ProgressSettings):
    """How much of the clip is transcribed."""

    main = progress_field("Clip")


@node(
    label="Transcribe",
    description="Turns the speech in an audio clip into text with a loaded model",
    menu="speech",
    search_tags=["speech", "stt", "asr", "transcribe", "text", "whisper", "dictation", "subtitle"],
    node_type=NodeType.DATA,
)
class TranscribeNode(BaseNode):
    """Turns the speech in an audio clip into text with a loaded model.

    Connect `Model` to Load Speech Model and `Audio` to a clip, such as Collect Clip's. Any channel
    count and sample rate will do: the clip is mixed to mono and resampled to the model's rate.

    - `Language`: `auto` detects it; a code such as `en` or `de` skips detection and is
      more reliable on short clips.

    `Transcript` holds the text, the language and the segments with their start and end
    times; `Text` is the text alone. Stopping the graph stops after the current segment.
    """

    progress = bag(TranscribeProgress)

    def init(self) -> None:
        from haybale_audio.types.audio import AUDIO

        from haybale_speech.types.stt_model import STT_MODEL
        from haybale_speech.types.transcript import TRANSCRIPT

        self.add(AUDIO.as_inlet("audio", label="Audio", description="The clip to transcribe"))
        self.add(STT_MODEL.as_inlet("model", label="Model", description="A loaded speech-to-text model"))
        self.add(
            STRING.as_config(
                "language",
                default="auto",
                label="Language",
                widget=MenuWidget.config(
                    properties={"options": list(LANGUAGES), "new_value_mode": "add-unique"}
                ),
                description="Language code of the speech; auto detects it",
            )
        )
        self.add(
            TRANSCRIPT.as_outlet("transcript", label="Transcript", description="Text, language and segments")
        )
        self.add(STRING.as_outlet("text", label="Text", description="The transcribed text"))

    def worker(
        self, context: ExecutionContext, audio: AudioBlock | None, model: SttModel | None, language: str
    ) -> None:
        if model is None:
            raise ValueError("no model — connect Load Speech Model")
        if audio is None:
            raise ValueError("no audio clip")
        samples = model_input(audio, model.sample_rate)
        wanted = (language or "auto").strip().lower()
        with self.progress.busy("main", "transcribing") as span:
            try:
                result = transcribe(
                    model,
                    samples,
                    language=None if wanted == "auto" else wanted,
                    cancel=context.cancel,
                    on_progress=lambda share, count: span.step(share, f"{count} segments"),
                )
            except Cancelled:
                return None
        self.out("transcript", result)
        self.out("text", result.text)
        return None
