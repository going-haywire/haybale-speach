# Speech model catalog

Each `.json` file here is one model catalog entry, registered as
`haybale-speech:catalog:<file name without .json>`.

- `whisper-small.json`: OpenAI's Whisper `small`, MIT licensed, as Systran's CTranslate2
  conversion for faster-whisper (`Systran/faster-whisper-small` on Hugging Face, pinned to
  commit `536b0662742c02347bc0e980a01041f333bce120`, checked on 2026-11-23). A whole-repo
  address: it resolves to the snapshot folder, which Load STT Model hands to faster-whisper.
- `kokoro-82m.json`: hexgrad's Kokoro-82M v1.0, Apache-2.0, as the ONNX export kokoro-onnx
  runs (`fastrtc/kokoro-onnx` on Hugging Face, pinned to commit
  `8d07950c9b6c87ce6809e9bba7bd494336217c2a`, checked on 2026-11-27; SHA-256 of
  `kokoro-v1.0.onnx` `7d5df8ec…a6c5`, of `voices-v1.0.bin` `bca610b8…bf7d`). The mirror's card
  says MIT; the weights are hexgrad's, under Apache-2.0, which this entry states. A whole-repo
  address limited to the model and the voices: it resolves to the snapshot folder, which Load
  Speech Model hands to kokoro-onnx.
