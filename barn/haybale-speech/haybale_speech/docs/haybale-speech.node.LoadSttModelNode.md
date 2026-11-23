# Load STT Model

`haybale-speech:node:LoadSttModelNode` · kind: node

Loads a speech-to-text model onto the CPU or GPU, for Transcribe

## Ports

| id | direction | type | description |
|---|---|---|---|
| `model` | inlet | `haybale-models:type:MODEL_REF` | Model to load: a catalog entry |
| `status.loaded` | config | `haywire-core:type:STRING` | Name, backend, device and compute type of the loaded model |
| `loaded` | outlet | `haybale-speech:type:STT_MODEL` | The loaded model |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `device` | settings | `None` | Where models run: auto (a CUDA GPU when there is one, else the CPU), cpu or cuda |
| `compute_type` | settings | `None` | Number format the model computes in: auto is int8 on the CPU and float16 on a GPU; a format the device cannot run falls back to float32 |
| `threads` | settings | `None` | Threads a model uses on the CPU; 0 lets the engine choose |
| `loaded` | status | `'< shows after the first run >'` | Name, backend, device and compute type of the loaded model |

## Notes

Loads a speech-to-text model onto the CPU or GPU, for Transcribe.

Pick the model on the card: a catalog entry, such as *Whisper small*. A model that is not on
this machine is an error until it is fetched in the Models editor. The model runs with
faster-whisper (CTranslate2), so it must be a CTranslate2 Whisper folder.

- `Device`: where the model runs. `auto` uses a CUDA GPU when there is one, else the CPU.
  A device this machine lacks runs on the CPU, with a note.
- `Compute type`: the number format. `auto` is `int8` on the CPU and `float16` on a GPU.
  A format the device cannot run falls back to `float32`, with a note.
- `CPU threads`: `0` lets the engine choose.

All three default to the library's *Speech* settings. Nodes that pick the same model with
the same settings share one loaded copy. The model is loaded when the graph runs and freed
when it stops.
