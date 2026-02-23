from typing import Sequence
import numpy as np


class BSpline:
    def __init__(
        self,
        t: Sequence[float],
        c: Sequence[float],
        d: int,
    ):
        """
        t = knots
        c = bspline coefficients / control points
        d = bspline degree
        """
        self.t = t
        self.c = c
        self.d = d
        assert self.is_valid()

    def is_valid(self) -> bool:
        """Check if the B-spline configuration is valid."""
        if len(self.t) != len(self.c) + self.d + 1:
            # n + d + 1 knots
            return False
        for i in range(len(self.t) - 1):
            # Each knot must be more than previous
            if self.t[i] > self.t[i + 1]:
                return False
        return True

    def bases(self, x: float, k: int, i: int) -> float:
        """
        Evaluate the B-spline basis function i, k at input position x.
        (Note that i, k start at 0.)
        """
        if k == 1:
            if self.t[i] <= x < self.t[i + 1]:
                return 1.0
            elif (x == self.t[-1]) and (self.t[i] <= x <= self.t[i + 1]):
                return 1.0
            else:
                return 0.0

        left = 0.0
        denom_left = self.t[i + k - 1] - self.t[i]
        if denom_left != 0:
            left = ((x - self.t[i]) / denom_left) * self.bases(x, k - 1, i)

        right = 0.0
        denom_right = self.t[i + k] - self.t[i + 1]
        if denom_right != 0:
            right = ((self.t[i + k] - x) / denom_right) * self.bases(x, k - 1, i + 1)

        return left + right

    def interp(self, x: float) -> float:
        """Evaluate the B-spline at input position x."""
        # https://personal.math.vt.edu/embree/math5466/lecture10.pdf
        order = self.d + 1
        return sum(self.c[i] * self.bases(x, order, i) for i in range(len(self.c)))


if __name__ == "__main__":
    t = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]  # clamped cubic knot vector
    c = [0.0, 1.0, 0.5, 0.8, 0.2]  # 5 control points
    d = 3  # cubic
    spline = BSpline(t, c, d)
    # now interpolate at some value
    value = 0.3
    spline.interp(value)