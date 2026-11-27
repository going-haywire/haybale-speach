# haybale-speech — component index (v0.0.2)

## node
- `haybale-speech:node:CollectClipNode` — Collect Clip — Gathers the audio blocks of a flow into one clip, sent on Stop  _tags: audio, clip, collect, gather, buffer, speech, record, segment_  **DEPRECATED**
- `haybale-speech:node:LoadSpeechModelNode` — Load Speech Model — Loads a speech model onto the CPU or GPU, for Transcribe, Speak or Pick Voice  _tags: model, load, speech, stt, tts, asr, whisper, kokoro, transcribe, speak, voice_
- `haybale-speech:node:LoadSttModelNode` — Load STT Model — Loads a speech-to-text model onto the CPU or GPU, for Transcribe  _tags: model, load, speech, stt, asr, whisper, faster-whisper, transcribe_  **DEPRECATED**
- `haybale-speech:node:PickVoiceNode` — Pick Voice — Picks one of a text-to-speech model's own voices, for Speak  _tags: speech, tts, voice, speaker, kokoro, pick_
- `haybale-speech:node:SpeakNode` — Speak — Turns text into an audio clip with a loaded text-to-speech model  _tags: speech, tts, speak, say, voice, text to speech, synthesize, kokoro, read aloud_
- `haybale-speech:node:SpeechSegmenterNode` — Speech Segmenter — Cuts a live audio flow into speech clips for Audio Events; replaced by Clip Segmenter  _tags: audio, speech, vad, voice, segment, split, pause, silence, live, microphone_  **DEPRECATED**
- `haybale-speech:node:TranscribeNode` — Transcribe — Turns the speech in an audio clip into text with a loaded model  _tags: speech, stt, asr, transcribe, text, whisper, dictation, subtitle_

## type
- `haybale-speech:type:SPEECH_MODEL` — Speech Model — A speech model loaded on a device, ready to run. Never saved with the graph.
- `haybale-speech:type:STT_MODEL` — STT Model — A speech-to-text model loaded on a device, ready to run. Never saved with the graph.
- `haybale-speech:type:TRANSCRIPT` — Transcript — The text of a clip, with its language and timed segments. Never saved with the graph.
- `haybale-speech:type:TTS_MODEL` — TTS Model — A text-to-speech model loaded on a device, ready to run. Never saved with the graph.
- `haybale-speech:type:VOICE` — Voice — A voice a text-to-speech model speaks with. Never saved with the graph.

## setting
- `haybale-speech:setting:SpeechSettings` — Speech — Device, compute type, threads and workers for models

## panel
- `haybale-speech:panel:SpeechSettingsPanel` — Speech — Device, compute type, threads and workers for speech models.
