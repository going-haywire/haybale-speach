# Speech model catalog

Each `.json` file here is one model catalog entry, registered as
`haybale-speech:catalog:<file name without .json>`.

- `whisper-small.json`: OpenAI's Whisper `small`, MIT licensed, as Systran's CTranslate2
  conversion for faster-whisper (`Systran/faster-whisper-small` on Hugging Face, pinned to
  commit `536b0662742c02347bc0e980a01041f333bce120`, checked on 2026-11-23). A whole-repo
  address: it resolves to the snapshot folder, which Load STT Model hands to faster-whisper.
