# Speech Segmenter

`haybale-speech:node:SpeechSegmenterNode` · kind: node

Cuts a live audio flow into speech clips at the pauses, for Transcribe

## Ports

| id | direction | type | description |
|---|---|---|---|
| `add` | inlet | `haybale-audio:type:AUDIO_EXEC` | Signal for sequencing the nodes of an audio flow |
| `audio` | inlet | `haybale-audio:type:AUDIO` | Block to listen to |
| `callbacks` | inlet | `haybale-core:type:POOLED` | Connect to the Audio Event nodes that get the clips. |
| `threshold` | config | `haywire-core:type:FLOAT` | Level (RMS) above which a block counts as speech |
| `hangover_ms` | config | `haywire-core:type:INT` | Pause that ends a segment |
| `preroll_ms` | config | `haywire-core:type:INT` | Audio kept from before the first loud block |
| `min_seconds` | config | `haywire-core:type:FLOAT` | Shortest speech sent; shorter is discarded |
| `max_seconds` | config | `haywire-core:type:FLOAT` | Longest segment; a longer one is cut there |
| `status` | config | `haywire-core:type:STRING` | Text data |
| `next` | outlet | `haybale-audio:type:AUDIO_EXEC` | Signal for sequencing the nodes of an audio flow |

## Notes

Listens to an audio flow and sends each stretch of speech as a clip: mono, 16 kHz.

A block is loud when its level (RMS) is above `Threshold`. The first loud block opens a
segment, together with the `Pre-roll` before it, so a quiet first sound is kept. A pause
of `Hang-over` ends the segment and the clip goes to every Audio Event listening on
`Subscribers`; the pause itself is left out. Speech shorter than `Min Seconds` (a click, a
cough) is discarded; a segment that reaches `Max Seconds` is sent at once and the next loud
block opens a new one. The status shows the level, which helps to set the threshold.

The clip arrives in a flow of its own, so transcribing it never holds up the audio flow,
however long the model takes:

```text
Device In → Audio Event → Speech Segmenter (Add)
Speech Segmenter (Subscribers) ← Audio Event → Print, with Transcribe's Text as the message
```

The listening Audio Event's `Audio` is the clip, its `Timestamp` the start of the speech in
seconds since the segmenter's first block, and its `Block Number` the count of clips sent.

Inputs:
    add: Feed the block on `audio` (wire it to an Audio Event's Block Ready)
    audio: The block
    callbacks: Pooled AUDIO_CALLBACK subscriptions from the Audio Event nodes that get the clips
    threshold: Level (RMS) above which a block counts as speech
    hangover_ms: Pause that ends a segment
    preroll_ms: Audio kept from before the first loud block
    min_seconds: Shortest speech sent; shorter is discarded
    max_seconds: Longest segment; a longer one is cut there

Outputs:
    next: Control flow after the block is taken
