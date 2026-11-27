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
https://github.com/going-haywire/haybale-speech/blob/v0.0.2/marketstall.toml
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
