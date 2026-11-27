"""The speech backends: which runtime module loads a model, by its catalog ``backend``.

Each backend is a module next to this one with ``BACKEND``, ``KIND`` (``stt`` or ``tts``),
``REQUIRES`` (the import that must work), ``resolve(device, compute_type)`` →
``(device, compute_type, notes)``, ``load(path, device, compute_type, threads, *, name, notes)``
and ``release(model)``. Load Speech Model picks the module from the resolved catalog variant,
or for a plain file from the kind its outlet takes.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class Backend:
    """One speech backend: its catalog name, the kind of model it loads and its runtime module."""

    name: str
    kind: str
    module: str
    requires: str

    def installed(self) -> bool:
        """Return whether the backend's package can be imported here."""
        return importlib.util.find_spec(self.requires) is not None

    def runtime(self) -> ModuleType:
        """Return the backend's runtime module.

        Raises:
            BackendMissing: The backend's package is not installed.
        """
        if not self.installed():
            raise BackendMissing(f"backend {self.name} is not installed (no {self.requires} module)")
        return importlib.import_module(self.module)


class BackendMissing(RuntimeError):
    """A model needs a backend this environment does not have."""


#: Every backend, the default of each kind first.
BACKENDS: tuple[Backend, ...] = (
    Backend("faster-whisper", "stt", "haybale_speech.runtime.whisper", "faster_whisper"),
    Backend("kokoro-onnx", "tts", "haybale_speech.runtime.kokoro", "kokoro_onnx"),
)


def by_name(name: str) -> Backend:
    """Return the backend called *name*.

    Raises:
        BackendMissing: No backend has that name.
    """
    for backend in BACKENDS:
        if backend.name == name:
            return backend
    raise BackendMissing(f"no speech backend is called {name}")


def of_kind(kind: str | None) -> list[Backend]:
    """Return the backends that load models of *kind*; every backend for ``None``."""
    return [b for b in BACKENDS if kind is None or b.kind == kind]


def default_for(kind: str | None) -> Backend | None:
    """Return the backend a plain file of *kind* loads with, or ``None`` when *kind* is ``None``."""
    found = of_kind(kind)
    return found[0] if kind is not None and found else None


def preference(kind: str | None) -> list[dict[str, str]]:
    """The catalog variants to prefer for *kind*: those of the installed backends, then any."""
    return [{"backend": b.name} for b in of_kind(kind) if b.installed()] + [{}]
