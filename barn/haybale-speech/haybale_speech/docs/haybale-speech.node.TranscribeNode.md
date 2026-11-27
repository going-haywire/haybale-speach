# Transcribe

`haybale-speech:node:TranscribeNode` · kind: node

Turns the speech in an audio clip into text with a loaded model

## Ports

| id | direction | type | description |
|---|---|---|---|
| `audio` | inlet | `haybale-audio:type:AUDIO` | The clip to transcribe |
| `model` | inlet | `haybale-speech:type:STT_MODEL` | A loaded speech-to-text model |
| `language` | config | `haywire-core:type:STRING` | Language code of the speech; auto detects it |
| `progress.main` | config | `haywire-core:type:PROGRESS` |  |
| `transcript` | outlet | `haybale-speech:type:TRANSCRIPT` | Text, language and segments |
| `text` | outlet | `haywire-core:type:STRING` | The transcribed text |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `main` | progress | `PROGRESS(active=False, fraction=None, label='')` |  |

## Notes

Turns the speech in an audio clip into text with a loaded model.

Connect `Model` to Load Speech Model and `Audio` to a clip, such as Collect Clip's. Any channel
count and sample rate will do: the clip is mixed to mono and resampled to the model's rate.

- `Language`: `auto` detects it; a code such as `en` or `de` skips detection and is
  more reliable on short clips.

`Transcript` holds the text, the language and the segments with their start and end
times; `Text` is the text alone. Stopping the graph stops after the current segment.
