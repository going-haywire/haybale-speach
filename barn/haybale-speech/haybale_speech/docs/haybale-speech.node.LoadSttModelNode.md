# Load STT Model

`haybale-speech:node:LoadSttModelNode` · kind: node

Loads a speech-to-text model onto the CPU or GPU, for Transcribe

> **Deprecated:** Load STT Model is replaced by Load Speech Model, which loads every kind of speech model

## Ports

| id | direction | type | description |
|---|---|---|---|
| `model` | inlet | `haybale-models:type:MODEL_REF` | Model to load: a catalog entry |
| `status.loaded` | config | `haywire-core:type:STRING` | Name, backend, device and compute type of the loaded model |
| `loaded` | outlet | `haybale-speech:type:STT_MODEL` | The loaded model |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `device` | settings | `None` | Where models run: auto (a CUDA GPU when one can run, else the CPU), cpu or cuda |
| `compute_type` | settings | `None` | Number format the model computes in: auto is int8 on the CPU and float16 on a GPU; a format the device cannot run falls back to float32 |
| `threads` | settings | `None` | Threads a model uses on the CPU; 0 lets the engine choose |
| `workers` | settings | `None` | Clips one model works on at once, for flows that share it (Clip Events set to Pooled); each run shares the CPU threads, so more than the cores allow is slower |
| `loaded` | status | `'< shows after the first run >'` | Name, backend, device and compute type of the loaded model |

## Notes

Loads a speech-to-text model onto the CPU or GPU, for Transcribe. Replaced by Load Speech Model.

It works as before, and graphs saved with it keep loading. New graphs use Load Speech
Model, which takes the kind of model the node it feeds needs.

Pick the model on the card: a catalog entry, such as *Whisper small*. The settings
`Device`, `Compute type`, `CPU threads` and `Workers` are those of Load Speech Model.
