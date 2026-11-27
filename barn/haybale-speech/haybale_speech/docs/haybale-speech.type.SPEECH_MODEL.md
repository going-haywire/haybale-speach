# Speech Model

`haybale-speech:type:SPEECH_MODEL` · kind: type

A speech model loaded on a device, ready to run. Never saved with the graph.

## Details

- **wire_type**: `haywire-core:wire:DataWire`
- **default**: `{'value': {}}`
- **color**: `#8a7ad9`

## Notes

A speech model loaded on a device, of any kind.

Transcribe takes one kind (`STT_MODEL`), Speak another (`TTS_MODEL`), so a Load Speech Model
wired to them becomes that kind and offers only the models that fit.
