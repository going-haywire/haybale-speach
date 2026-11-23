"""SpeechSettings — library-wide device, compute type and threads for speech models."""

from haywire.api import CHOICES, INT, LibrarySettings, setting, settings

from haybale_speech.runtime.devices import COMPUTE_TYPES, DEVICES


@settings(namespace="speech", label="Speech", description="Device, compute type and threads for models")
class SpeechSettings(LibrarySettings):
    """Where and how speech models run; Load STT Model can change each one per node."""

    device = setting[CHOICES](
        "auto",
        label="Device",
        description="Where models run: auto (a CUDA GPU when there is one, else the CPU), cpu or cuda",
        widget_config={"options": list(DEVICES)},
    )
    compute_type = setting[CHOICES](
        "auto",
        label="Compute type",
        description=(
            "Number format the model computes in: auto is int8 on the CPU and float16 on a GPU; "
            "a format the device cannot run falls back to float32"
        ),
        widget_config={"options": list(COMPUTE_TYPES)},
    )
    threads = setting[INT](
        0,
        label="CPU threads",
        description="Threads a model uses on the CPU; 0 lets the engine choose",
        min=0,
        max=256,
    )
