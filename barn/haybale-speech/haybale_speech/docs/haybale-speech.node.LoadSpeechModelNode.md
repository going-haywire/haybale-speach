# Load Speech Model

`haybale-speech:node:LoadSpeechModelNode` · kind: node

Loads a speech model onto the CPU or GPU, for Transcribe, Speak or Pick Voice

## Ports

| id | direction | type | description |
|---|---|---|---|
| `status.loaded` | config | `haywire-core:type:STRING` | Name, backend, device and compute type of the loaded model |
| `loaded` | outlet | `haywire-core:type:ADD` | Connect it to the node that runs the model; it takes that node's kind of model |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `device` | settings | `None` | Where models run: auto (a CUDA GPU when one can run, else the CPU), cpu or cuda |
| `compute_type` | settings | `None` | Number format the model computes in: auto is int8 on the CPU and float16 on a GPU; a format the device cannot run falls back to float32 |
| `threads` | settings | `None` | Threads a model uses on the CPU; 0 lets the engine choose |
| `workers` | settings | `None` | Clips one model works on at once, for flows that share it (Clip Events set to Pooled); each run shares the CPU threads, so more than the cores allow is slower |
| `loaded` | status | `'< shows after the first run >'` | Name, backend, device and compute type of the loaded model |

## Notes

Loads a speech model onto the CPU or GPU, for Transcribe, Speak or Pick Voice.

Connect the `Model` outlet first, to the node that runs the model: the outlet takes the
kind of model that node needs (speech-to-text for Transcribe, text-to-speech for Speak and
Pick Voice), and the `Model` picker appears, offering only models of that kind. The outlet
keeps its kind until the node is reset. Wired only to nodes outside the family, the
outlet stays open and the node loads nothing; the first speech node wired to it gives it
its kind.

Pick the model on the card: a catalog entry, such as *Whisper small* or *Kokoro 82M*. A
model that is not on this machine is an error until it is fetched in the Models editor.
The catalog entry names the backend that runs it (faster-whisper, kokoro-onnx); a backend
that is not installed is an error on the card.

- `Device`: where the model runs. `auto` uses a CUDA GPU when one can run, else the CPU;
  a GPU that cannot run (no cuBLAS for CUDA 12, say) gives a note. A device this machine
  lacks runs on the CPU, with a note; `cuda` on a GPU that cannot run is an error that
  says why.
- `Compute type`: the number format, for faster-whisper models. `auto` is `int8` on the CPU
  and `float16` on a GPU. A format the device cannot run falls back to `float32`, with a
  note. kokoro-onnx runs the precision of the model file, and notes any other choice.
- `CPU threads`: `0` lets the engine choose.
- `Workers`: how many clips the model works on at once, for flows that share it, such as
  Clip Events set to Pooled behind one Clip Segmenter. `1` runs one at a time. More only
  helps while cores are idle: each run shares the CPU threads. faster-whisper gets it as
  `num_workers` (CTranslate2's `inter_threads`), which lets concurrent calls run in parallel
  on one loaded model at the cost of more memory per worker; kokoro-onnx needs no setting
  of its own.

All four default to the library's *Speech* settings. Nodes that pick the same model with
the same settings share one loaded copy. The model is loaded when the graph runs and freed
when it stops.
