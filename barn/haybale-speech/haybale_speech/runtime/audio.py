"""Turn an audio clip into the samples a speech model takes: one channel at its sample rate."""

from __future__ import annotations

import numpy as np
from haybale_audio.dsp.channels import to_mono
from haybale_audio.value import AudioBlock


def model_input(clip: AudioBlock, sample_rate: int) -> np.ndarray:
    """Return *clip* as a 1-D float32 array, mixed to mono and resampled to *sample_rate*.

    The whole clip is resampled at once, so its end is not cut off as a streaming
    resampler would.
    """
    mono = to_mono(clip).data[0]
    if clip.sample_rate == sample_rate or mono.size == 0:
        return np.ascontiguousarray(mono, dtype=np.float32)
    import samplerate

    out = samplerate.resample(mono, sample_rate / clip.sample_rate, "sinc_best")
    return np.ascontiguousarray(out, dtype=np.float32)
