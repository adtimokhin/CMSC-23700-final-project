import sys
import os
import numpy as np
from nodes.base import Node

# Add project root so we can import interpolation.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interpolation import BSpline


class SmoothingNode(Node):
    """
    Smooths audio feature arrays using B-spline interpolation.
    Downsamples the signal to control points, then re-evaluates at all frames.
    """

    def __init__(
        self,
        fields: list,
        n_control_points: int = 20,
        degree: int = 3,
        name: str = None,
    ):
        """
        fields: which data keys to smooth, e.g. ["audio_volume", "audio_bass"]
        n_control_points: fewer = smoother
        degree: B-spline degree (3 = cubic)
        """
        super().__init__(name)
        self.fields = fields
        self.n_control_points = n_control_points
        self.degree = degree

    def process(self, data: dict) -> dict:
        n_frames = data["n_frames"]

        for field in self.fields:
            if field not in data:
                continue
            signal = data[field]

            # Downsample to control points
            indices = np.linspace(0, len(signal) - 1, self.n_control_points).astype(int)
            control_points = signal[indices].tolist()

            # Build clamped knot vector
            n_cp = self.n_control_points
            d = self.degree
            n_knots = n_cp + d + 1
            n_interior = n_knots - 2 * (d + 1)
            if n_interior > 0:
                interior = np.linspace(0, 1, n_interior + 2)[1:-1].tolist()
            else:
                interior = []
            knots = [0.0] * (d + 1) + interior + [1.0] * (d + 1)

            spline = BSpline(knots, control_points, d)

            # Evaluate at each frame
            xs = np.linspace(0, 1, n_frames)
            data[field] = np.array([spline.interp(x) for x in xs])

        return data
