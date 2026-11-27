"""Load Speech Model — load a picked speech model onto a device, of the kind the node it feeds takes."""

from __future__ import annotations

from typing import Any

from haybale_models.consumer import models_state, release_node, resolve_port
from haybale_models.value.model_ref import ModelRef
from haywire.api import (
    ADD,
    STRING,
    BaseNode,
    ExecutionContext,
    LabelWidget,
    NodeSettings,
    NodeType,
    NodeWarning,
    PortType,
    Promotable,
    bag,
    node,
    setting,
    shadow,
)

from haybale_speech.runtime.backends import Backend, BackendMissing, by_name, default_for, preference
from haybale_speech.settings.speech_settings import SpeechSettings

#: Warning kind of the device and compute type notes on the card.
KIND = "speech"

#: The card line while no model is loaded.
NOT_LOADED = "< shows after the first run >"


class LoadSpeechModelSettings(NodeSettings):
    """Device, compute type and threads of one Load Speech Model node."""

    device = shadow(SpeechSettings.device)
    compute_type = shadow(SpeechSettings.compute_type)
    threads = shadow(SpeechSettings.threads)


class LoadSpeechModelStatus(NodeSettings):
    """What the node loaded, shown on its card."""

    loaded = setting[STRING](
        NOT_LOADED,
        label="Loaded",
        description="Name, backend, device and compute type of the loaded model",
        widget=LabelWidget.config(),
        promotable=Promotable.CONFIG,
        promote_default=PortType.CONFIG,
    )


class BaseSpeechLoaderNode(BaseNode):
    """What every speech model loader shares: resolve, pick the backend, load into the pool, release.

    A subclass gives the outlet its kind (``_outlet_kind``) and adds the ports.
    """

    settings = bag(LoadSpeechModelSettings)
    status = bag(LoadSpeechModelStatus)

    def _outlet_kind(self) -> Any:
        """The model type the outlet carries, or ``None`` while it is open."""
        raise NotImplementedError

    def _add_model_inlet(self, kind: Any) -> None:
        """Add the model picker, offering the catalog entries of *kind*."""
        from haybale_models.types.model_ref import MODEL_REF

        properties = {"filter": dict(kind.catalog_filter or {}), "prefer": preference(kind.purpose)}
        self.add(
            MODEL_REF.as_inlet(
                "model",
                label="Model",
                widget_config={"properties": properties},
                description="Model to load: a catalog entry",
            )
        )

    def post_init(self) -> None:
        self.settings._subscribe(self._on_settings_changed)

    def _on_settings_changed(self, name: str, value: Any, old: Any) -> None:
        """Another device, compute type or thread count needs another load: run at the next execution."""
        self.run_again()

    def worker(self, context: ExecutionContext, model: ModelRef) -> None:
        kind = self._outlet_kind()
        if kind is None:
            raise ValueError("connect the Model outlet first: it decides which kind of model to load")
        try:
            resolved = resolve_port(self, "model")
        except Exception:
            self._let_go()
            raise
        name = resolved.entry.name if resolved.entry is not None else resolved.path.name
        backend = _backend_for(resolved, kind, name)
        try:
            runtime = backend.runtime()
        except BackendMissing:
            self._let_go()
            raise
        device, compute_type, notes = runtime.resolve(
            str(self.settings.device), str(self.settings.compute_type)
        )
        if not compute_type and resolved.resource is not None:
            compute_type = resolved.resource.precision or ""
        threads = int(self.settings.threads)
        key = (resolved.key, (backend.name, device, compute_type, threads))
        if getattr(self.cache, "key", None) != key:
            self._let_go()
        loaded = models_state().pool.acquire(
            key,
            self,
            load=lambda: runtime.load(resolved.path, device, compute_type, threads, name=name, notes=notes),
            release=runtime.release,
        )
        self.cache.key = key
        if not kind.fits(loaded):
            self._let_go()
            raise ValueError(f"{name} does not fit this outlet ({kind.class_identity.label.lower()})")
        self.wrapper.replace_warnings(
            KIND, [NodeWarning(n, source_version=None, kind=KIND) for n in loaded.notes]
        )
        self.status.loaded = loaded.status()
        self.out("loaded", loaded)
        return None

    def on_shutdown(self, context: ExecutionContext) -> None:
        self._let_go()
        # Drop the wire's reference too, so the model's memory is freed while the graph is stopped.
        self.out("loaded", None)
        self.run_again()

    def on_teardown(self) -> None:
        self._let_go()

    def _let_go(self) -> None:
        """Release this node's hold on its loaded model, if any."""
        self.cache.key = None
        try:
            release_node(self)
        except RuntimeError:
            pass  # haybale-models already disabled: its pool released everything
        try:
            self.status.loaded = NOT_LOADED
        except Exception:  # noqa: BLE001
            pass

    def model_preference(self, port_id: str) -> Any:
        """The catalog variants this node prefers: those of the installed backends of its kind."""
        kind = self._outlet_kind()
        return preference(kind.purpose if kind is not None else None)


def _backend_for(resolved: Any, kind: Any, name: str) -> Backend:
    """The backend that loads *resolved* for an outlet of *kind*.

    The catalog variant names it; a plain file loads with the default backend of the kind.

    Raises:
        ValueError: The model is of another kind, or a plain file has no kind to go by.
    """
    named = resolved.resource.backend if resolved.resource is not None else None
    if named:
        backend = by_name(named)
    else:
        found = default_for(kind.purpose)
        if found is None:
            raise ValueError(
                f"{name} names no backend — pick a catalog entry, or wire Model to Transcribe or Speak"
            )
        backend = found
    if kind.purpose is not None and backend.kind != kind.purpose:
        raise ValueError(
            f"{name} is a {backend.kind} model, which does not fit this outlet "
            f"({kind.class_identity.label.lower()})"
        )
    return backend


@node(
    label="Load Speech Model",
    description="Loads a speech model onto the CPU or GPU, for Transcribe, Speak or Pick Voice",
    menu="speech",
    search_tags=[
        "model",
        "load",
        "speech",
        "stt",
        "tts",
        "asr",
        "whisper",
        "kokoro",
        "transcribe",
        "speak",
        "voice",
    ],
    node_type=NodeType.DATA,
)
class LoadSpeechModelNode(BaseSpeechLoaderNode):
    """Loads a speech model onto the CPU or GPU, for Transcribe, Speak or Pick Voice.

    Connect the `Model` outlet first, to the node that runs the model: the outlet takes the
    kind of model that node needs (speech-to-text for Transcribe, text-to-speech for Speak and
    Pick Voice), and the `Model` picker appears, offering only models of that kind. The outlet
    keeps its kind until the node is reset.

    Pick the model on the card: a catalog entry, such as *Whisper small* or *Kokoro 82M*. A
    model that is not on this machine is an error until it is fetched in the Models editor.
    The catalog entry names the backend that runs it (faster-whisper, kokoro-onnx); a backend
    that is not installed is an error on the card.

    - `Device`: where the model runs. `auto` uses a CUDA GPU when there is one, else the CPU.
      A device this machine lacks runs on the CPU, with a note.
    - `Compute type`: the number format, for faster-whisper models. `auto` is `int8` on the CPU
      and `float16` on a GPU. A format the device cannot run falls back to `float32`, with a
      note. kokoro-onnx runs the precision of the model file, and notes any other choice.
    - `CPU threads`: `0` lets the engine choose.

    All three default to the library's *Speech* settings. Nodes that pick the same model with
    the same settings share one loaded copy. The model is loaded when the graph runs and freed
    when it stops.
    """

    def init(self) -> None:
        self.add(
            ADD.as_outlet(
                "loaded",
                label="Model",
                description="Connect it to the node that runs the model; it takes that node's kind of model",
                on_connect="_on_outlet_connected",
            )
        )

    def _on_outlet_connected(self, port: Any, edge_wrapper: Any) -> None:
        """Give the open outlet the kind of model the connected inlet takes, once.

        An inlet outside the speech model family makes it ``SPEECH_MODEL``, so that edge shows
        as invalid instead of carrying a model where none fits.
        """
        from haybale_speech.types.speech_model import SPEECH_MODEL

        if port.type_cls is not ADD:
            return
        sink = edge_wrapper._inlet_port.stored_type
        kind = sink if isinstance(sink, type) and issubclass(sink, SPEECH_MODEL) else SPEECH_MODEL
        with self.rejig(include=[port.id, "model"]):
            self._add_model_inlet(kind)
            self.add(
                kind.as_outlet(
                    port.id,
                    label="Model",
                    description=f"The loaded model: {kind.class_identity.label.lower()}",
                    on_connect="_on_outlet_connected",
                )
            )
        self.wrapper.redraw()

    def _outlet_kind(self) -> Any:
        from haybale_speech.types.speech_model import SPEECH_MODEL

        kind = self.ports["loaded"].type_cls
        return kind if isinstance(kind, type) and issubclass(kind, SPEECH_MODEL) else None
