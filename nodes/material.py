import numpy as np
from nodes.base import Node


class MaterialNode(Node):
    """Maps audio features to per-frame material color changes."""

    def __init__(
        self,
        obj_name: str,
        source: str = "audio_volume",
        color_low=(0.0, 0.0, 1.0),
        color_high=(1.0, 0.0, 0.0),
        name: str = None,
    ):
        super().__init__(name)
        self.obj_name = obj_name
        self.source = source
        self.color_low = np.array(color_low)
        self.color_high = np.array(color_high)

    def validate(self, data: dict):
        if "objects" not in data or self.obj_name not in data["objects"]:
            raise ValueError(
                f"MaterialNode requires '{self.obj_name}' in data['objects']. "
                "Add an ObjectTransformNode for this object first."
            )

    def process(self, data: dict) -> dict:
        n = data["n_frames"]
        audio = data[self.source]

        colors = np.zeros((n, 3))
        for f in range(n):
            t = audio[f]
            colors[f] = (1 - t) * self.color_low + t * self.color_high

        data["objects"][self.obj_name]["material_colors"] = colors
        return data
