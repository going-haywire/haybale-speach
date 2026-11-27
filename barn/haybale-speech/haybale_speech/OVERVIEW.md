# Speech

Speech-to-text and text-to-speech, with faster-whisper and Kokoro

## Nodes
### Speech
- **Collect Clip** — Gathers the audio blocks of a flow into one clip, sent on Stop
- **Load STT Model** — Loads a speech-to-text model onto the CPU or GPU, for Transcribe
- **Load Speech Model** — Loads a speech model onto the CPU or GPU, for Transcribe, Speak or Pick Voice
- **Pick Voice** — Picks one of a text-to-speech model's own voices, for Speak
- **Speak** — Turns text into an audio clip with a loaded text-to-speech model
- **Speech Segmenter** — Cuts a live audio flow into speech clips for Audio Events; replaced by Clip Segmenter
- **Transcribe** — Turns the speech in an audio clip into text with a loaded model

## Types
- **STT Model** — A speech-to-text model loaded on a device, ready to run. Never saved with the graph.
- **Speech Model** — A speech model loaded on a device, ready to run. Never saved with the graph.
- **TTS Model** — A text-to-speech model loaded on a device, ready to run. Never saved with the graph.
- **Transcript** — The text of a clip, with its language and timed segments. Never saved with the graph.
- **Voice** — A voice a text-to-speech model speaks with. Never saved with the graph.

## Settings
- **Speech** — Device, compute type, threads and workers for models

## Panels
- **Speech** — Device, compute type, threads and workers for speech models.
