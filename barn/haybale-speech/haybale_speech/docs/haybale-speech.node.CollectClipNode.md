# Collect Clip

`haybale-speech:node:CollectClipNode` · kind: node

Gathers the audio blocks of a flow into one clip, sent on Stop

> **Deprecated:** This Collect Clip moved to haybale-audio (menu audio/clip); use that one

## Ports

| id | direction | type | description |
|---|---|---|---|
| `add` | inlet | `haybale-audio:type:AUDIO_EXEC` | Signal for sequencing the nodes of an audio flow |
| `stop` | inlet | `haywire-core:type:EXEC` | Signal for controlling execution flow between nodes |
| `audio` | inlet | `haybale-audio:type:AUDIO` | Block to append |
| `max_seconds` | config | `haywire-core:type:FLOAT` | Longest clip kept; later blocks are dropped |
| `status` | config | `haywire-core:type:STRING` | Text data |
| `next` | outlet | `haybale-audio:type:AUDIO_EXEC` | Signal for sequencing the nodes of an audio flow |
| `done` | outlet | `haywire-core:type:EXEC` | Runs after Stop, once Clip is set |
| `clip` | outlet | `haybale-audio:type:AUDIO_CLIP` | The gathered audio |
| `seconds` | outlet | `haywire-core:type:FLOAT` | Length of the clip so far |

## Notes

haybale-audio's Collect Clip under its 0.0.2 name in haybale-speech. Replaced by that node.

It works as haybale-audio's does, and graphs saved with it keep loading: its `Clip` is an
Audio Clip, so it still connects to Transcribe. In 0.0.2 `Clip` was Audio, a block: an
edge saved from it to an Audio inlet, such as a Recorder's or Reblock's, now loads broken
(no adapter from Audio Clip to Audio). Play the clip through Clip Player into an Audio
Event instead. New graphs use Collect Clip from the *audio/clip* menu.
