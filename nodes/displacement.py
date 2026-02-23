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

    def process(self, data: dict) -> dict:
        obj_data = data["objects"][self.obj_name]
        obj_file = obj_data["obj_file"]
        n_frames = data["n_frames"]

        vertices, normals = _parse_obj_vertices_and_normals(obj_file)
        n_verts = len(vertices)

        audio = data[self.source]  # shape (n_frames,), values in [0, 1]

        # offsets[f, v] = how far vertex v moves at frame f, as an (x, y, z) vector.
        # Multiplying the normal by a scalar pushes the vertex outward (positive)
        # or inward (negative) along the surface — keeps deformation smooth.
        offsets = np.zeros((n_frames, n_verts, 3))
        for f in range(n_frames):
            # Every vertex gets the same audio-driven scale at a given frame,
            # so the whole mesh "breathes" uniformly in and out.
            offsets[f] = normals * audio[f] * self.amplitude

        obj_data["vertex_offsets"] = offsets
        # base_vertices is the rest pose; Blender adds offsets on top of these.
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
            # "v" lines define vertex positions; "vn" lines define per-vertex normals.
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vn":
                normals.append([float(x) for x in parts[1:4]])

    vertices = np.array(vertices)

    if len(normals) > 0:
        normals = np.array(normals)
        # OBJ normal count doesn't always equal vertex count (normals can be
        # per-face-corner). If they mismatch, fall back to treating vertex
        # position as the outward direction — works well for convex meshes like spheres.
        if len(normals) != len(vertices):
            normals = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8)
    else:
        # No normals in the file at all — same position-as-normal fallback.
        # +1e-8 prevents division by zero for any vertex sitting at the origin.
        normals = vertices / (np.linalg.norm(vertices, axis=1, keepdims=True) + 1e-8)

    return vertices, normals
