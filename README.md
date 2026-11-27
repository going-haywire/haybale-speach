# Haybale Speech

A haywire project.

## Subscribe

<!-- marketstall:share-url:start -->
Always the latest (tracks the current branch):

```sh
https://github.com/going-haywire/haybale-speech/blob/master/marketstall.toml
```

Frozen to this version:

```sh
https://github.com/going-haywire/haybale-speech/blob/v0.0.3/marketstall.toml
```
<!-- marketstall:share-url:end -->

## Licences

The library is MIT. Its text-to-speech backend, kokoro-onnx (MIT), turns text into
phonemes with [phonemizer](https://github.com/bootphon/phonemizer) and the espeak-ng library
that `espeakng-loader` ships; both are **GPL-3.0**. They are runtime dependencies, installed
from PyPI, not part of this repository. The models have their own licences, stated in each
catalog entry (`barn/haybale-speech/haybale_speech/catalog/`): Whisper small is MIT,
Kokoro 82M is Apache-2.0.

## Getting started

```sh
uv sync
uv run haywire
```
## Demos

Two graphs in `examples/`; open them in the studio. Both listen to the default input device,
cut it into one clip per sentence with Clip Segmenter (haybale-audio) and transcribe each clip
with Whisper small.

- `live_speech_to_text.haywire` prints what it hears.
- `speech_round_trip.haywire` also speaks it back with Kokoro 82M and plays it on the default
  output device: Device In → Audio Event → Clip Segmenter → Clip Event → Transcribe → Speak →
  Clip Player → Audio Event → Device Out. **Use headphones**: through speakers the microphone
  hears the reply and the circle never ends. A reply that comes while the last one plays cuts
  it off.

Both models are catalog entries; fetch them in the Models editor first.

## Share your library

`haywire share` releases the project: it bumps every file to the version, writes
`marketstall.toml`, updates the subscribe link above, then **commits, tags and
pushes by itself**. Run it from the project root with a clean working tree
(`git checkout uv.lock` first if `uv` re-locked it). The version is an explicit
`x.y.z` or `major`/`minor`/`patch`. Look first, then release:
```sh
uv run --frozen haywire share --dry-run
uv run --frozen haywire share --bump patch
```

See [CLAUDE.md](CLAUDE.md) for the release rules.
