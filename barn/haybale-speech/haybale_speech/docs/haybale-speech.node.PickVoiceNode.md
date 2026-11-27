# Pick Voice

`haybale-speech:node:PickVoiceNode` · kind: node

Picks one of a text-to-speech model's own voices, for Speak

## Ports

| id | direction | type | description |
|---|---|---|---|
| `model` | inlet | `haybale-speech:type:TTS_MODEL` | A loaded text-to-speech model |
| `voice_name` | config | `haywire-core:type:STRING` | The model's voice |
| `voice` | outlet | `haybale-speech:type:VOICE` | The voice, for Speak |
| `language` | outlet | `haywire-core:type:STRING` | The language the voice speaks |

## Notes

Picks one of a text-to-speech model's own voices, for Speak.

Connect `Model` to Load Speech Model. `Voice Name` is one of the model's voices, such as
`af_heart`; a name the model lacks is an error that lists the voices it has. `Language` is
the language the voice speaks.
