"""Library settings panel for the Speech library, on the ``BarnsSurface`` surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from haywire.api.workbench import BarnsSurface, BasePanel, PanelLayout, hui, panel, render_schema

from haybale_speech.settings.speech_settings import SpeechSettings

if TYPE_CHECKING:
    from haywire.api.workbench import SessionContext


@panel(
    surface=BarnsSurface,
    label="Speech",
    description="Device, compute type, threads and workers for speech models.",
    icon=hui.icon.models,
    order=20,
    default_open=False,
)
class SpeechSettingsPanel(BasePanel):
    """Device, compute type, threads and workers for speech models."""

    def draw(self, ctx: "SessionContext", layout: PanelLayout) -> None:
        registry = ctx.app.library_service.get_settings_registry()
        render_schema(ctx, SpeechSettings, registry)
