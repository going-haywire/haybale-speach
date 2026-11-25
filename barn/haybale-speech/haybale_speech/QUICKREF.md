# haybale-speech — component index (v0.0.1)

## node
- `haybale-speech:node:CollectClipNode` — Collect Clip — Gathers the audio blocks of a flow into one clip, sent on Stop  _tags: audio, clip, collect, gather, buffer, speech, record, segment_
- `haybale-speech:node:LoadSttModelNode` — Load STT Model — Loads a speech-to-text model onto the CPU or GPU, for Transcribe  _tags: model, load, speech, stt, asr, whisper, faster-whisper, transcribe_
- `haybale-speech:node:SpeechSegmenterNode` — Speech Segmenter — Cuts a live audio flow into speech clips at the pauses, for Transcribe  _tags: audio, speech, vad, voice, segment, split, pause, silence, live, microphone_
- `haybale-speech:node:TranscribeNode` — Transcribe — Turns the speech in an audio clip into text with a loaded model  _tags: speech, stt, asr, transcribe, text, whisper, dictation, subtitle_

## type
- `haybale-speech:type:STT_MODEL` — STT Model — A speech-to-text model loaded on a device, ready to run. Never saved with the graph.
- `haybale-speech:type:TRANSCRIPT` — Transcript — The text of a clip, with its language and timed segments. Never saved with the graph.

## setting
- `haybale-speech:setting:SpeechSettings` — Speech — Device, compute type and threads for models

## panel
- `haybale-speech:panel:SpeechSettingsPanel` — Speech — Device, compute type and threads for speech models.
