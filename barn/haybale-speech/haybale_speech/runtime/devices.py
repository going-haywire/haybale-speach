"""Turn the Device and Compute type settings into what CTranslate2 accepts."""

from __future__ import annotations

#: Devices CTranslate2 runs on; ``auto`` uses a CUDA GPU when there is one.
DEVICES = ("auto", "cpu", "cuda")

#: Number formats CTranslate2 computes in; ``auto`` is int8 on the CPU, float16 on a GPU.
COMPUTE_TYPES = ("auto", "int8", "int8_float32", "int8_float16", "int16", "float16", "bfloat16", "float32")


def resolve_device(name: str, cuda_devices: int | None = None) -> tuple[str, str | None]:
    """Return the CTranslate2 device for a Device setting, and a note when it falls back to the CPU.

    Args:
        name: ``auto``, ``cpu`` or ``cuda``; blank reads as ``auto``.
        cuda_devices: CUDA devices on this machine; ``None`` asks CTranslate2.
    """
    wanted = (name or "auto").strip().lower()
    if cuda_devices is None:
        import ctranslate2

        cuda_devices = ctranslate2.get_cuda_device_count()
    if wanted == "auto":
        return ("cuda" if cuda_devices > 0 else "cpu"), None
    if wanted == "cpu":
        return "cpu", None
    if wanted == "cuda":
        if cuda_devices > 0:
            return "cuda", None
        return "cpu", "cuda not available — running on cpu"
    return "cpu", f"{name} is not a device — running on cpu"


def resolve_compute_type(
    name: str, device: str, supported: set[str] | None = None
) -> tuple[str, str | None]:
    """Return the compute type to load with, and a note when the choice falls back.

    ``auto`` is ``int8`` on the CPU and ``float16`` on CUDA. A choice *device* does not
    support runs ``float32``.

    Args:
        name: One of :data:`COMPUTE_TYPES`; blank reads as ``auto``.
        device: ``cpu`` or ``cuda``.
        supported: The compute types *device* supports; ``None`` asks CTranslate2.
    """
    wanted = (name or "auto").strip().lower()
    if supported is None:
        import ctranslate2

        supported = set(ctranslate2.get_supported_compute_types(device))
    if wanted == "auto":
        wanted = "float16" if device == "cuda" else "int8"
        if wanted in supported:
            return wanted, None
        return "float32", None
    if wanted not in COMPUTE_TYPES:
        return "float32", f"{name} is not a compute type — running float32"
    if wanted not in supported:
        return "float32", f"{wanted} not supported on {device} — running float32"
    return wanted, None
