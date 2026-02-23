import numpy as np
from nodes.base import Node


class ObjectTransformNode(Node):
    """
    Maps audio features to object transforms (location, rotation, scale).
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

        # np.tile repeats the 1-D base vector n times to get an (n, 3) array,
        # one row per frame. This is the "no animation" starting point.
        locations = np.tile(self.base_location, (n, 1))
        rotations = np.tile(self.base_rotation, (n, 1))
        scales = np.tile(self.base_scale, (n, 1))

        for prop, cfg in self.mapping.items():
            source = data[cfg["source"]]  # shape (n_frames,), values in [0, 1]
            lo, hi = cfg["range"]
            # Linearly map the [0, 1] audio signal into [lo, hi].
            values = lo + source * (hi - lo)

            if prop == "scale":
                # axis_mask selects which axes get animated (e.g. [1,0,1] = X and Z only).
                # The expression below leaves unmasked axes at their current value
                # while replacing masked axes with the audio-driven value.
                axis_mask = np.array(cfg.get("axis", [1, 1, 1]), dtype=float)
                scales *= (1 - axis_mask[None, :]) + values[:, None] * axis_mask[None, :]
            elif prop.startswith("location_"):
                # prop[-1] is "x", "y", or "z"; "xyz".index() converts it to 0/1/2.
                idx = "xyz".index(prop[-1])
                locations[:, idx] += values
            elif prop.startswith("rotation_"):
                idx = "xyz".index(prop[-1])
                rotations[:, idx] += values

        if "objects" not in data:
            data["objects"] = {}

        # Store everything this object needs under its name so later nodes
        # (MaterialNode, NoiseDisplacementNode, etc.) can find it by obj_name.
        data["objects"][self.obj_name] = {
            "obj_file": self.obj_file,
            "locations": locations,
            "rotations": rotations,
            "scales": scales,
            # Default grey material; MaterialNode will overwrite material_colors.
            "material": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0},
        }

        return data
