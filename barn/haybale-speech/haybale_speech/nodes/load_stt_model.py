"""Load STT Model — load a picked speech-to-text model onto a device, for Transcribe."""

from __future__ import annotations

from typing import Any

from haybale_models.consumer import models_state, release_node, resolve_port
from haybale_models.value.model_ref import ModelRef
from haywire.api import (
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

from haybale_speech.runtime.devices import resolve_compute_type, resolve_device
from haybale_speech.runtime.whisper import BACKEND, load_whisper, release_whisper
from haybale_speech.settings.speech_settings import SpeechSettings

#: Warning kind of the device and compute type notes on the card.
KIND = "speech"

#: The card line while no model is loaded.
NOT_LOADED = "< shows after the first run >"

#: Variants Load STT Model prefers from a catalog entry: the faster-whisper ones.
PREFER = [{"backend": BACKEND}, {}]


class LoadSttModelSettings(NodeSettings):
    """Device, compute type and threads of one Load STT Model node."""

    device = shadow(SpeechSettings.device)
    compute_type = shadow(SpeechSettings.compute_type)
    threads = shadow(SpeechSettings.threads)


class LoadSttModelStatus(NodeSettings):
    """What the node loaded, shown on its card."""

    loaded = setting[STRING](
        NOT_LOADED,
        label="Loaded",
        description="Name, backend, device and compute type of the loaded model",
        widget=LabelWidget.config(),
        promotable=Promotable.CONFIG,
        promote_default=PortType.CONFIG,
    )


@node(
    label="Load STT Model",
    description="Loads a speech-to-text model onto the CPU or GPU, for Transcribe",
    menu="speech",
    search_tags=["model", "load", "speech", "stt", "asr", "whisper", "faster-whisper", "transcribe"],
    node_type=NodeType.DATA,
)
class LoadSttModelNode(BaseNode):
    """Loads a speech-to-text model onto the CPU or GPU, for Transcribe.

    Pick the model on the card: a catalog entry, such as *Whisper small*. A model that is not on
    this machine is an error until it is fetched in the Models editor. The model runs with
    faster-whisper (CTranslate2), so it must be a CTranslate2 Whisper folder.

    - `Device`: where the model runs. `auto` uses a CUDA GPU when one can run, else the CPU;
      a GPU that cannot run (no cuBLAS for CUDA 12, say) gives a note. A device this machine
      lacks runs on the CPU, with a note; `cuda` on a GPU that cannot run is an error that
      says why.
    - `Compute type`: the number format. `auto` is `int8` on the CPU and `float16` on a GPU.
      A format the device cannot run falls back to `float32`, with a note.
    - `CPU threads`: `0` lets the engine choose.

    All three default to the library's *Speech* settings. Nodes that pick the same model with
    the same settings share one loaded copy. The model is loaded when the graph runs and freed
    when it stops.
    """

    settings = bag(LoadSttModelSettings)
    status = bag(LoadSttModelStatus)

    def init(self) -> None:
        from haybale_models.types.model_ref import MODEL_REF

        from haybale_speech.types.stt_model import STT_MODEL

        properties = {"filter": dict(STT_MODEL.catalog_filter or {}), "prefer": PREFER}
        self.add(
            MODEL_REF.as_inlet(
                "model",
                label="Model",
                widget_config={"properties": properties},
                description="Model to load: a catalog entry",
            )
        )
        self.add(STT_MODEL.as_outlet("loaded", label="Model", description="The loaded model"))

    def post_init(self) -> None:
        self.settings._subscribe(self._on_settings_changed)

    def _on_settings_changed(self, name: str, value: Any, old: Any) -> None:
        """Another device, compute type or thread count needs another load: run at the next execution."""
        self.run_again()

    def worker(self, context: ExecutionContext, model: ModelRef) -> None:
        device, device_note = resolve_device(str(self.settings.device))
        compute_type, compute_note = resolve_compute_type(str(self.settings.compute_type), device)
        threads = int(self.settings.threads)
        notes = tuple(n for n in (device_note, compute_note) if n)
        try:
            resolved = resolve_port(self, "model")
        except Exception:
            self._let_go()
            raise
        key = (resolved.key, (BACKEND, device, compute_type, threads))
        if getattr(self.cache, "key", None) != key:
            self._let_go()
        name = resolved.entry.name if resolved.entry is not None else resolved.path.name
        loaded = models_state().pool.acquire(
            key,
            self,
            load=lambda: load_whisper(resolved.path, device, compute_type, threads, name=name, notes=notes),
            release=release_whisper,
        )
        self.cache.key = key
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
        """The catalog variants this node prefers: the faster-whisper ones."""
        return PREFER
