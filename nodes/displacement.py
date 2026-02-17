import numpy as np
from nodes.base import Node


class VertexDisplacementNode(Node):
    """
    Computes per-vertex, per-frame offsets that push vertices along their normals
    based on an audio feature. Applied in Blender via shape keys.
    """

    def __init__(
        self,
        obj_name: str,
        source: str = "audio_bass",
        amplitude: float = 0.3,
        name: str = None,
    ):
        """
        obj_name: must match a key in data["objects"]
        source: audio feature field to drive displacement
        amplitude: max displacement distance along normal
        """
        super().__init__(name)
        self.obj_name = obj_name
        self.source = source
        self.amplitude = amplitude

    def validate(self, data: dict):
        if "objects" not in data or self.obj_name not in data["objects"]:
            raise ValueError(
                f"VertexDisplacementNode requires '{self.obj_name}' in data['objects']. "
                "Add an ObjectTransformNode for this object first."
            )

    def process(self, data: dict) -> dict:
        obj_data = data["objects"][self.obj_name]
        obj_file = obj_data["obj_file"]
        n_frames = data["n_frames"]

        vertices, normals = _parse_obj_vertices_and_normals(obj_file)
        n_verts = len(vertices)

        audio = data[self.source]  # shape (n_frames,)

        # offsets shape: (n_frames, n_verts, 3)
        offsets = np.zeros((n_frames, n_verts, 3))
        for f in range(n_frames):
            offsets[f] = normals * audio[f] * self.amplitude

        obj_data["vertex_offsets"] = offsets
        obj_data["base_vertices"] = vertices

        return data


def _parse_obj_vertices_and_normals(obj_file: str):
    """Simple OBJ parser for vertices and normals."""
    vertices = []
    normals = []
    with open(obj_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vn":
                normals.append([float(x) for x in parts[1:4]])

    vertices = np.array(vertices)

    if len(normals) > 0:
        normals = np.array(normals)
        if len(normals) != len(vertices):
            # Fall back to position-based normals for convex meshes
            normals = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8)
    else:
        normals = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8)

    return vertices, normals
