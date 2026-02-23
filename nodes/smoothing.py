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

            # Pick n_control_points evenly-spaced samples from the raw signal.
            # These become the B-spline's control points. Fewer points = smoother
            # curve because high-frequency jitter gets averaged away.
            indices = np.linspace(0, len(signal) - 1, self.n_control_points).astype(int)
            control_points = signal[indices].tolist()

            # Build a "clamped" knot vector: (d+1) copies of 0 at the start and
            # (d+1) copies of 1 at the end force the spline to pass through the
            # first and last control points rather than just approximating them.
            n_cp = self.n_control_points
            d = self.degree
            n_knots = n_cp + d + 1
            # Interior knots fill the gap between the clamped ends. If the
            # math gives 0 interior knots (happens when n_cp is small) we skip them.
            n_interior = n_knots - 2 * (d + 1)
            if n_interior > 0:
                interior = np.linspace(0, 1, n_interior + 2)[1:-1].tolist()
            else:
                interior = []
            knots = [0.0] * (d + 1) + interior + [1.0] * (d + 1)

            spline = BSpline(knots, control_points, d)

            # Re-evaluate the spline at n_frames evenly-spaced positions in [0, 1].
            # This gives us a smooth version of the original signal at the same
            # length as the other per-frame arrays in data.
            xs = np.linspace(0, 1, n_frames)
            data[field] = np.array([spline.interp(x) for x in xs])

        return data
