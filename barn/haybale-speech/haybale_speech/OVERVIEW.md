# Speech

Speech-to-text for audio clips, with faster-whisper

## Nodes
### Speech
- **Collect Clip** — Gathers the audio blocks of a flow into one clip, sent on Stop
- **Load STT Model** — Loads a speech-to-text model onto the CPU or GPU, for Transcribe
- **Speech Segmenter** — Cuts a live audio flow into speech clips at the pauses, for Transcribe
- **Transcribe** — Turns the speech in an audio clip into text with a loaded model

## Types
- **STT Model** — A speech-to-text model loaded on a device, ready to run. Never saved with the graph.
- **Transcript** — The text of a clip, with its language and timed segments. Never saved with the graph.

## Settings
- **Speech** — Device, compute type and threads for models

## Panels
- **Speech** — Device, compute type and threads for speech models.
