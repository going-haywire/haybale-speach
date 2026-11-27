"""Collect Clip — the speech copy of 0.0.2, kept so saved graphs load. Use haybale-audio's Collect Clip."""

from __future__ import annotations

from haybale_audio.nodes.collect_clip import CollectClipNode as AudioCollectClipNode
from haywire.api import NodeType, node


@node(
    label="Collect Clip",
    description="Gathers the audio blocks of a flow into one clip, sent on Stop",
    menu="speech",
    search_tags=["audio", "clip", "collect", "gather", "buffer", "speech", "record", "segment"],
    node_type=NodeType.CONTROL,
    deprecation_warning="This Collect Clip moved to haybale-audio (menu audio/clip); use that one",
)
class CollectClipNode(AudioCollectClipNode):
    """haybale-audio's Collect Clip under its 0.0.2 name in haybale-speech. Replaced by that node.

    It works as haybale-audio's does, and graphs saved with it keep loading: its `Clip` is an
    Audio Clip, so it still connects to Transcribe. New graphs use Collect Clip from the
    *audio/clip* menu.
    """
