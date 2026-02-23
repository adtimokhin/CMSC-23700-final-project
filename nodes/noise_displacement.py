import numpy as np
from nodes.base import Node
from nodes.displacement import _parse_obj_vertices_and_normals


class NoiseDisplacementNode(Node):
    """
    Displaces vertices along their normals using a spatially-varying noise pattern
    that evolves over time. The overall amplitude is modulated by an audio feature.

    Each vertex gets a noise value based on its position in space + time, which
    determines both direction (positive/negative along normal) and per-vertex amplitude.
    The audio source (e.g. volume) scales the entire effect.
    """

    def __init__(
        self,
        obj_name: str,
        source: str = "audio_volume",
        amplitude: float = 0.3,
        noise_scale: float = 2.0,
        time_speed: float = 1.0,
        octaves: int = 3,
        seed: int = 42,
        name: str = None,
    ):
        """
        obj_name: must match a key in data["objects"]
        source: audio feature field that modulates amplitude
        amplitude: max displacement distance along normal
        noise_scale: spatial frequency of the noise (higher = more detail)
        time_speed: how fast the noise pattern evolves over time
        octaves: number of noise layers (more = more detail)
        seed: random seed for reproducibility
        """
        super().__init__(name)
        self.obj_name = obj_name
        self.source = source
        self.amplitude = amplitude
        self.noise_scale = noise_scale
        self.time_speed = time_speed
        self.octaves = octaves
        self.seed = seed

    def validate(self, data: dict):
        if "objects" not in data or self.obj_name not in data["objects"]:
            raise ValueError(
                f"NoiseDisplacementNode requires '{self.obj_name}' in data['objects']. "
                "Add an ObjectTransformNode for this object first."
            )

    def process(self, data: dict) -> dict:
        obj_data = data["objects"][self.obj_name]
        obj_file = obj_data["obj_file"]
        n_frames = data["n_frames"]
        time = data["time"]
        audio = data[self.source]

        vertices, normals = _parse_obj_vertices_and_normals(obj_file)
        n_verts = len(vertices)

        # Generate random frequency vectors for each octave (seeded for reproducibility)
        rng = np.random.default_rng(self.seed)
        freq_vectors = []
        for _ in range(self.octaves):
            # 3 spatial frequencies + 1 time frequency + 1 phase offset per octave
            freq_vectors.append({
                "spatial": rng.normal(0, 1, size=(3,)),
                "time_freq": rng.normal(0, 1),
                "phase": rng.uniform(0, 2 * np.pi),
            })

        # Compute noise for each vertex at each frame
        # offsets shape: (n_frames, n_verts, 3)
        offsets = np.zeros((n_frames, n_verts, 3))

        for f in range(n_frames):
            t = time[f] * self.time_speed

            # Accumulate noise across octaves
            noise_values = np.zeros(n_verts)
            for octave_idx, fv in enumerate(freq_vectors):
                freq_scale = 2.0 ** octave_idx
                octave_weight = 0.5 ** octave_idx

                # Dot product of vertex positions with spatial frequency vector
                spatial_dot = (vertices * self.noise_scale * freq_scale) @ fv["spatial"]
                # Add time component
                noise_values += octave_weight * np.sin(
                    spatial_dot + t * fv["time_freq"] + fv["phase"]
                )

            # noise_values is in roughly [-1, 1] range per vertex
            # This determines direction and per-vertex amplitude
            # Scale by audio volume and max amplitude
            offsets[f] = normals * (noise_values[:, None] * audio[f] * self.amplitude)

        obj_data["vertex_offsets"] = offsets
        obj_data["base_vertices"] = vertices

        return data
