from nodes.base import Node
from pipeline.pipeline import Pipeline


class ExportNode(Node):
    """Terminal node: saves frame data to disk as a manifest for the Blender render script."""

    def __init__(self, output_dir: str, name: str | None = None):
        super().__init__(name)
        self.output_dir = output_dir

    def process(self, data: dict) -> dict:
        # audio_raw is a large waveform array that Blender doesn't need.
        # Dropping it keeps the manifest files small and fast to load.
        export_data = {k: v for k, v in data.items() if k != "audio_raw"}
        Pipeline.save_manifest(export_data, self.output_dir)
        return data
