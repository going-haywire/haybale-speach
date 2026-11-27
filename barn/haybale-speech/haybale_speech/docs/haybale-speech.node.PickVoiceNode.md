# Pick Voice

`haybale-speech:node:PickVoiceNode` · kind: node

Picks one of a text-to-speech model's own voices, for Speak

## Ports

| id | direction | type | description |
|---|---|---|---|
| `model` | inlet | `haybale-speech:type:TTS_MODEL` | A loaded text-to-speech model |
| `settings.voice_name` | config | `haywire-core:type:STRING` | The model's voice, such as af_heart; the first letter of a Kokoro voice is its language |
| `voice` | outlet | `haybale-speech:type:VOICE` | The voice, for Speak |
| `language` | outlet | `haywire-core:type:STRING` | The language the voice speaks |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `voice_name` | settings | `'af_heart'` | The model's voice, such as af_heart; the first letter of a Kokoro voice is its language |

## Notes

Picks one of a text-to-speech model's own voices, for Speak.

Connect `Model` to Load Speech Model. `Voice Name` is one of the model's voices, such as
`af_heart`; a name the model lacks is an error that lists the voices it has. `Language` is
the language the voice speaks.

`Voice Name` is a setting, shown on the card. Promote it to an inlet to choose the voice from
another node while the graph runs, or to an outlet to tell other nodes which voice speaks. A
new name is picked at the next run.
