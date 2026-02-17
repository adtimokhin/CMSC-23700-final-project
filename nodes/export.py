from nodes.base import Node
from pipeline.pipeline import Pipeline


class ExportNode(Node):
    """Terminal node: saves frame data to disk as a manifest for the Blender render script."""

    def __init__(self, output_dir: str, name: str = None):
        super().__init__(name)
        self.output_dir = output_dir

    def process(self, data: dict) -> dict:
        # Strip raw audio from export (large, not needed by Blender)
        export_data = {k: v for k, v in data.items() if k != "audio_raw"}
        Pipeline.save_manifest(export_data, self.output_dir)
        print(f"[Export] Manifest saved to {self.output_dir}")
        return data
