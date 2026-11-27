"""Pick Voice — one of a text-to-speech model's own voices, for Speak."""

from __future__ import annotations

from haywire.api import STRING, BaseNode, ExecutionContext, MenuWidget, NodeType, node

from haybale_speech.nodes.speak import VOICES
from haybale_speech.runtime.kokoro import voice as model_voice
from haybale_speech.value import TtsModel


@node(
    label="Pick Voice",
    description="Picks one of a text-to-speech model's own voices, for Speak",
    menu="speech",
    search_tags=["speech", "tts", "voice", "speaker", "kokoro", "pick"],
    node_type=NodeType.DATA,
)
class PickVoiceNode(BaseNode):
    """Picks one of a text-to-speech model's own voices, for Speak.

    Connect `Model` to Load Speech Model. `Voice Name` is one of the model's voices, such as
    `af_heart`; a name the model lacks is an error that lists the voices it has. `Language` is
    the language the voice speaks.
    """

    def init(self) -> None:
        from haybale_speech.types.tts_model import TTS_MODEL
        from haybale_speech.types.voice import VOICE

        self.add(TTS_MODEL.as_inlet("model", label="Model", description="A loaded text-to-speech model"))
        self.add(
            STRING.as_config(
                "voice_name",
                default=VOICES[0],
                label="Voice Name",
                widget=MenuWidget.config(
                    properties={"options": list(VOICES), "new_value_mode": "add-unique"}
                ),
                description="The model's voice",
                on_change="_on_voice_name_changed",
            )
        )
        self.add(VOICE.as_outlet("voice", label="Voice", description="The voice, for Speak"))
        self.add(STRING.as_outlet("language", label="Language", description="The language the voice speaks"))

    def _on_voice_name_changed(self, port: object, value: object) -> None:
        """Another voice: pick it at the next execution, while the graph runs too.

        A config write marks no port dirty, and the model inlet rarely changes, so without this
        the node would keep handing out the voice it picked first.
        """
        self.run_again()

    def worker(self, context: ExecutionContext, model: TtsModel | None, voice_name: str) -> None:
        if model is None:
            raise ValueError("no model — connect Load Speech Model")
        picked = model_voice(model, voice_name)
        self.out("voice", picked)
        self.out("language", picked.language)
        return None
