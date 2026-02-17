import numpy as np
from nodes.base import Node


class ObjectTransformNode(Node):
    """
    Maps audio features to object transforms (location, rotation, scale).

    mapping example:
    {
        "scale": {
            "source": "audio_volume",
            "axis": [1, 1, 1],
            "range": [0.5, 2.0],
        },
        "rotation_z": {
            "source": "audio_bass",
            "range": [0.0, 6.28],
        },
        "location_z": {
            "source": "audio_onset",
            "range": [0.0, 1.5],
        },
    }
    """

    def __init__(
        self,
        obj_name: str,
        obj_file: str,
        mapping: dict,
        base_location=(0, 0, 0),
        base_rotation=(0, 0, 0),
        base_scale=(1, 1, 1),
        name: str = None,
    ):
        super().__init__(name)
        self.obj_name = obj_name
        self.obj_file = obj_file
        self.mapping = mapping
        self.base_location = np.array(base_location, dtype=float)
        self.base_rotation = np.array(base_rotation, dtype=float)
        self.base_scale = np.array(base_scale, dtype=float)

    def process(self, data: dict) -> dict:
        n = data["n_frames"]

        locations = np.tile(self.base_location, (n, 1))
        rotations = np.tile(self.base_rotation, (n, 1))
        scales = np.tile(self.base_scale, (n, 1))

        for prop, cfg in self.mapping.items():
            source = data[cfg["source"]]  # shape (n_frames,), range [0,1]
            lo, hi = cfg["range"]
            values = lo + source * (hi - lo)

            if prop == "scale":
                axis_mask = np.array(cfg.get("axis", [1, 1, 1]), dtype=float)
                scales *= (1 - axis_mask[None, :]) + values[:, None] * axis_mask[None, :]
            elif prop.startswith("location_"):
                idx = "xyz".index(prop[-1])
                locations[:, idx] += values
            elif prop.startswith("rotation_"):
                idx = "xyz".index(prop[-1])
                rotations[:, idx] += values

        if "objects" not in data:
            data["objects"] = {}

        data["objects"][self.obj_name] = {
            "obj_file": self.obj_file,
            "locations": locations,
            "rotations": rotations,
            "scales": scales,
            "material": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
        }

        return data
