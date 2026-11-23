"""Haybale-Speech: speech-to-text for audio clips.

Component folders (registered in ``register_components``):
  - ``settings``, ``types``, ``nodes``, ``catalog`` and the scaffold's other folders.

``runtime`` holds the faster-whisper code the nodes call; it loads the engine only when a
model runs.
"""

from pathlib import Path

from haywire.api import (
    AdapterRegistry,
    BaseLibrary,
    CatalogRegistry,
    MacroRegistry,
    NodeRegistry,
    SettingsRegistry,
    SkinRegistry,
    ThemeRegistry,
    TypeRegistry,
    WidgetRegistry,
    library,
)
from haywire.api.workbench import EditorRegistry, StateRegistry, PanelRegistry


@library(
    file_watcher=True,
)
class Library(BaseLibrary):
    """Speech library — load speech models and turn audio clips into text."""

    def register_components(self):
        """Register all components with the global registries."""
        base_path = Path(__file__).parent

        self.add_folder_to_registry(
            folder_path=str(base_path / "settings"),
            registry_cls=SettingsRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "states"),
            registry_cls=StateRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "themes"),
            registry_cls=ThemeRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "types"),
            registry_cls=TypeRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "adapters"),
            registry_cls=AdapterRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "widgets"),
            registry_cls=WidgetRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "skins"),
            registry_cls=SkinRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "nodes"),
            registry_cls=NodeRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "macros"),
            registry_cls=MacroRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "catalog"),
            registry_cls=CatalogRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "panels"),
            registry_cls=PanelRegistry,
        )

        self.add_folder_to_registry(
            folder_path=str(base_path / "editors"),
            registry_cls=EditorRegistry,
        )

    def validate(self) -> bool:
        """Validate library structure."""
        return True
