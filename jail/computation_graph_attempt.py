from typing import List, Set, Dict, Tuple
from collections.abc import Iterator, Callable
import numpy as np

# a class for a differential computation graph
# based in part on https://github.com/davidrosenberg/mlcourse/blob/gh-pages/Notebooks/computation-graph/computation-graph-framework.ipynb

class Value(object):
    """This computation graph node stores a value, that is, an object without inputs/parents.
    These can be input data or parameters in layers."""

    def __init__(self, name: str, data: np.array) -> None:
        self.name = name
        self.data = data
        self.d_data = None
        self.inputs = []

    def forward(self) -> np.array:
        self.d_data = np.zeros_like(self.data)
        return self.data
    
    def vjp_fun(self, vec: np.array) -> np.array:
        return np.zeros_like(vec)

    def backward(self) -> np.array:
        return self.d_data
    
class Dot(object):
    """This node computes a np.dot(. , .) operation on the inputs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.data = None
        self.d_data = None
        self.inputs = []

    def forward(self, lhs, rhs ) -> np.array:
        self.inputs = [lhs, rhs] # store references to input nodes
        self.data = np.dot(lhs.data, rhs.data)
        self.d_data = np.zeros_like(self.data)
        return self.data
    
    def vjp_fun(self, input, mult, vec):
        # The VJP w.r.t. input of np.dot(input, mult)
        return np.dot(vec, mult)

    def backward(self):
        # Preconditions: self.d_data contains the partial derivatives of the graph output w.r.t. self.data
        # and was already updated by its children
        # accumulate gradient into inputs
        lhs, rhs = self.inputs[0], self.inputs[1]
        d_lhs = self.vjp_fun(lhs, rhs, self.d_data)
        d_rhs = self.vjp_fun(rhs.T, lhs.T, self.d_data)
        lhs.d_data += d_lhs
        rhs.d_data += d_rhs

        return self.d_data