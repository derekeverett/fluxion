from typing import List, Optional
import numpy as np

# a class for a differential computation graph
# based in part on https://github.com/davidrosenberg/mlcourse/blob/gh-pages/Notebooks/computation-graph/computation-graph-framework.ipynb


class Dual:
    """A class for scalar dual numbers."""

    def __init__(self, v: float, d: float):
        self.v = v
        self.d = d

    def __add__(self, other: "Dual"):
        return Dual(self.v + other.v, self.d + other.d)

    def __sub__(self, other: "Dual"):
        return Dual(self.v - other.v, self.d - other.d)

    def __mul__(self, other: "Dual"):
        return Dual(self.v * other.v, self.v * other.d + self.d * other.v)

    def __pow__(self, beta: float):
        return Dual(self.v**beta, beta * self.d * (self.v**beta - 1.0))
