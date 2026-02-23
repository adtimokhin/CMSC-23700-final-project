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


    def process(self, data: dict) -> dict:
        obj_data = data["objects"][self.obj_name]
        obj_file = obj_data["obj_file"]
        n_frames = data["n_frames"]
        time = data["time"]
        audio = data[self.source]

        vertices, normals = _parse_obj_vertices_and_normals(obj_file)
        n_verts = len(vertices)

        # Each "octave" is an independent noise layer with a random spatial direction
        # and a random time oscillation frequency. We seed the RNG so the pattern
        # is identical across runs (deterministic render).
        rng = np.random.default_rng(self.seed)
        freq_vectors = []
        for _ in range(self.octaves):
            freq_vectors.append({
                # spatial: a 3-D vector; dotting it with a vertex position gives a
                # scalar that varies continuously across the mesh surface.
                "spatial": rng.normal(0, 1, size=(3,)),
                # time_freq: how fast this octave oscillates over time.
                "time_freq": rng.normal(0, 1),
                # phase: shifts the sine wave so octaves don't all peak simultaneously.
                "phase": rng.uniform(0, 2 * np.pi),
            })

        offsets = np.zeros((n_frames, n_verts, 3))

        for f in range(n_frames):
            t = time[f] * self.time_speed

            # Sum contributions from each octave (fractal / Perlin-like noise).
            # Higher octaves have 2x the frequency and half the weight — each layer
            # adds finer detail without overwhelming the coarser structure below.
            noise_values = np.zeros(n_verts)
            for octave_idx, fv in enumerate(freq_vectors):
                freq_scale = 2.0 ** octave_idx   # doubles spatial frequency each octave
                octave_weight = 0.5 ** octave_idx  # halves amplitude each octave

                # Project all vertex positions onto this octave's spatial direction.
                # Result is a (n_verts,) array — each entry is a unique phase offset
                # per vertex, making neighbouring vertices slightly out of sync.
                spatial_dot = (vertices * self.noise_scale * freq_scale) @ fv["spatial"]

                # sin() maps the phase to [-1, 1], giving direction as well as magnitude.
                # Adding the time term makes the pattern animate over time.
                noise_values += octave_weight * np.sin(
                    spatial_dot + t * fv["time_freq"] + fv["phase"]
                )

            # noise_values[:, None] broadcasts to (n_verts, 3) so each vertex's
            # scalar noise value scales all three components of its normal vector.
            # audio[f] gates the whole effect: silence → no displacement.
            offsets[f] = normals * (noise_values[:, None] * audio[f] * self.amplitude)

        obj_data["vertex_offsets"] = offsets
        obj_data["base_vertices"] = vertices

        return data
