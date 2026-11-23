"""Fixtures for this repo's own suite (mirrors haybale-spandrel's conftest).

The installed haywire libraries (haybale-core, -audio, -models, ...) load through their entry
points, this repo's library from ``barn/``. Only the tests marked ``slow`` touch a real model.
"""

from __future__ import annotations

import os
import pwd
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from haywire.core.volumes import VolumeStore, get_volume_store, set_volume_store

#: The real home, read before the suite points HOME at a temp folder (for the Hugging Face cache).
REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """The repo root (the directory holding ``barn/``)."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def _isolated_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point HOME at a temp dir so library enables never write into the real ``~/.haywire``."""
    home = tmp_path_factory.mktemp("home")
    previous = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    yield home
    if previous is not None:
        os.environ["HOME"] = previous


@pytest.fixture(scope="session")
def library_system(project_root: Path, _isolated_home: Path) -> Iterator[object]:
    """A fully initialised library system with this repo's library loaded."""
    from haywire.core.di.config import set_global_injector, set_library_system
    from haywire.core.di.test_config import create_test_library_system

    service = create_test_library_system(
        workspace_root=str(project_root),
        library_paths=[str(project_root / "barn")],
        load_libraries=True,
        enable_file_watching=False,
    )
    set_library_system(service)
    set_global_injector(service.injector)

    yield service

    lib_registry = service.get_library_registry()
    if hasattr(lib_registry, "stop_file_watching"):
        lib_registry.stop_file_watching()
    set_library_system(None)
    set_global_injector(None)


@pytest.fixture
def graph(library_system: Any) -> Any:
    """An empty graph on the loaded library system, validating inline."""
    from haywire.api.workbench import BaseGraph
    from haywire.core.graph.scheduler import SyncScheduler

    return BaseGraph(filestem="Speech Test Graph", validation_scheduler=SyncScheduler())


@pytest.fixture
def models_env(tmp_path: Path) -> Iterator[VolumeStore]:
    """A temp VolumeStore with both model volumes pointed at tmp_path subfolders."""
    from haybale_models.keys import HF_VOLUME, MODELS_VOLUME

    previous = get_volume_store()
    store = VolumeStore(tmp_path / "models-volumes.toml")
    set_volume_store(store)
    store.ensure(MODELS_VOLUME, label="Models", description="", location=tmp_path / "models")
    store.ensure(HF_VOLUME, label="Hugging Face cache", description="", location=tmp_path / "hf")
    (tmp_path / "models").mkdir()
    (tmp_path / "hf").mkdir()
    yield store
    set_volume_store(previous)


@pytest.fixture(scope="session")
def speech_wav(project_root: Path) -> Path:
    """About 3 s of synthetic speech, 16 kHz mono: "The quick brown fox jumps over the lazy dog." """
    return project_root / "tests/fixtures/quick-brown-fox.wav"


def add_node(graph: Any, module: str, name: str) -> Any:
    """Place the node class *name* from *module* in *graph* and return the node."""
    import importlib

    cls = getattr(importlib.import_module(module), name)
    wrapper = graph.create_node_wrapper(cls.class_identity.registry_key, position=(0, 0))
    assert wrapper is not None, f"node creation failed for {name}"
    assert isinstance(wrapper.node, cls), "an unknown key yields an ErrorNode"
    return wrapper.node


def run_from(graph: Any, entry: Any) -> None:
    """Assemble *graph* and run the flow that starts at the event node *entry*, startup to shutdown."""
    import threading

    from haywire.core.assembly.flow_assembly_manager import FlowAssemblyManager
    from haywire.core.execution.event_source import Trigger
    from haywire.core.execution.vm import HaywireVM

    graph.force_validation()
    flows = FlowAssemblyManager().assemble_graph(graph)
    flow = next(f for f in flows if f.get_entry_node_id() == entry.node_id)
    trigger = Trigger(source_key=flow.get_subscription_key(), payload={}, timestamp=0.0)
    vm = HaywireVM()
    nodes = list(flow.get_all_nodes())

    def run() -> None:
        vm.call_startup(nodes, flow)
        try:
            vm.execute_control_flow(flow, trigger)
        finally:
            vm.call_shutdown(nodes, flow)

    runner = threading.Thread(target=run)
    runner.start()
    runner.join()


def place_model(source: Path, name: str | None = None) -> Any:
    """Copy *source* into the Models volume, index it, and return a pinned ``ModelRef``."""
    import hashlib
    import shutil

    from haybale_models.consumer import models_state
    from haybale_models.keys import MODELS_VOLUME
    from haybale_models.value.model_ref import ModelRef
    from haywire.api import volume_location

    target = volume_location(MODELS_VOLUME) / (name or source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    state = models_state()
    state.index.scan()
    state.index.hash_pending()
    data = target.read_bytes()
    return ModelRef(sha256=hashlib.sha256(data).hexdigest(), size=len(data), label=target.name)


@pytest.fixture(autouse=True)
def _empty_pool(request: pytest.FixtureRequest) -> Iterator[None]:
    """Release every pooled model after a test that loaded the library system."""
    yield
    if "library_system" in request.fixturenames:
        from haybale_models.consumer import models_state

        models_state().pool.release_all()
