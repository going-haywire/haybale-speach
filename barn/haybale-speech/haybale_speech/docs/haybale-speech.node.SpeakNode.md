# Speak

`haybale-speech:node:SpeakNode` · kind: node

Turns text into an audio clip with a loaded text-to-speech model

## Ports

| id | direction | type | description |
|---|---|---|---|
| `text` | inlet | `haywire-core:type:STRING` | The text to speak |
| `model` | inlet | `haybale-speech:type:TTS_MODEL` | A loaded text-to-speech model |
| `voice` | inlet | `haybale-speech:type:VOICE` | A voice from Pick Voice; replaces Voice Name |
| `voice_name` | config | `haywire-core:type:STRING` | The model's voice to speak with, while no Voice is wired |
| `speed` | config | `haywire-core:type:FLOAT` | 0.5 to 2.0; 1.0 is the voice's own pace |
| `progress.main` | config | `haywire-core:type:PROGRESS` |  |
| `audio` | outlet | `haybale-audio:type:AUDIO` | The spoken clip, mono |
| `seconds` | outlet | `haywire-core:type:FLOAT` | Length of the clip |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `main` | progress | `PROGRESS(active=False, fraction=None, label='')` |  |

## Notes

Turns text into an audio clip with a loaded text-to-speech model.

Connect `Model` to Load Speech Model and `Text` to the text to speak, such as Transcribe's
`Text`. The clip is mono, at the model's sample rate (24 kHz for Kokoro).

- `Voice Name`: the voice by name, such as `af_heart` (American English) or `bf_emma` (British
  English); the first letter of a Kokoro voice is its language. A `Voice` wire, from Pick
  Voice, takes its place.
- `Speed`: 0.5 to 2.0; 1.0 is the voice's own pace.

`Audio` is the spoken clip and `Seconds` its length. The text is spoken one sentence at a
time; stopping the graph stops after the current sentence.
