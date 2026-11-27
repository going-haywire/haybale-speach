"""Turn the Device and Compute type settings into what CTranslate2 accepts."""

from __future__ import annotations

import ctypes
import functools
import sys
from collections.abc import Callable

#: Devices CTranslate2 runs on; ``auto`` uses a CUDA GPU when one can run, else the CPU.
DEVICES = ("auto", "cpu", "cuda")

#: Number formats CTranslate2 computes in; ``auto`` is int8 on the CPU, float16 on a GPU.
COMPUTE_TYPES = ("auto", "int8", "int8_float32", "int8_float16", "int16", "float16", "bfloat16", "float32")

#: What :func:`cuda_problem` answers on a machine without a CUDA device.
NO_CUDA_DEVICE = "no CUDA device"

#: The cuBLAS library CTranslate2 4 loads on first use of a GPU, per platform.
CUBLAS = {"linux": "libcublas.so.12", "win32": "cublas64_12.dll"}


@functools.cache
def cuda_problem() -> str | None:
    """Return why CUDA cannot run on this machine, or ``None`` when it can.

    Checked once per process: CTranslate2 must see a CUDA device, and the cuBLAS library it
    loads lazily, only when a model first computes, must load. Without that check a model
    loads on the GPU and fails at its first transcription.
    """
    import ctranslate2

    if ctranslate2.get_cuda_device_count() == 0:
        return NO_CUDA_DEVICE
    library = CUBLAS.get(sys.platform)
    if library is not None:
        try:
            ctypes.CDLL(library)
        except OSError:
            return f"{library} is not found (cuBLAS for CUDA 12)"
    return None


def resolve_device(name: str, probe: Callable[[], str | None] = cuda_problem) -> tuple[str, str | None]:
    """Return the CTranslate2 device for a Device setting, and a note when it falls back to the CPU.

    ``auto`` runs on CUDA when *probe* finds nothing wrong, else on the CPU, with a note when
    the machine has a CUDA device that cannot run.

    Args:
        name: ``auto``, ``cpu`` or ``cuda``; blank reads as ``auto``.
        probe: Says why CUDA cannot run, or ``None`` when it can; see :func:`cuda_problem`.

    Raises:
        RuntimeError: *name* is ``cuda`` and the machine has a CUDA device that cannot run.
    """
    wanted = (name or "auto").strip().lower()
    if wanted == "cpu":
        return "cpu", None
    if wanted not in ("auto", "cuda"):
        return "cpu", f"{name} is not a device — running on cpu"
    problem = probe()
    if problem is None:
        return "cuda", None
    if wanted == "auto":
        if problem == NO_CUDA_DEVICE:
            return "cpu", None
        return "cpu", f"cuda cannot run: {problem} — running on cpu"
    if problem == NO_CUDA_DEVICE:
        return "cpu", "cuda not available — running on cpu"
    raise RuntimeError(f"cuda cannot run: {problem}; pick cpu or auto as the Device")


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
