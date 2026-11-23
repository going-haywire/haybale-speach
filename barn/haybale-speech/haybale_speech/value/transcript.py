"""Transcript — the text of a clip, with its language and timed segments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A stretch of speech: start and end in seconds from the start of the clip, and its text."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """What a speech-to-text model heard in one clip.

    ``text`` joins the segments' texts with spaces. ``language`` is the code the model was
    told or detected (``en``, ``de``, ...); ``duration`` is the clip's length in seconds.
    """

    text: str
    language: str
    segments: tuple[Segment, ...] = ()
    duration: float = 0.0

    @classmethod
    def from_segments(cls, segments: list[Segment], language: str, duration: float) -> Transcript:
        """Build a transcript whose text joins *segments* in order."""
        text = " ".join(s.text for s in segments if s.text)
        return cls(text=text, language=language, segments=tuple(segments), duration=duration)
