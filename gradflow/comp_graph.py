from typing import List, Optional
import numpy as np

# a class for a differential computation graph
# based in part on https://github.com/davidrosenberg/mlcourse/blob/gh-pages/Notebooks/computation-graph/computation-graph-framework.ipynb

class Node:
    """A base class for computation graph nodes."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.out: Optional[np.array] = None
        self.d_out: Optional[np.array] = None
        self.inputs: List['GraphNode'] = []

    def forward(self, *args, **kwargs) -> np.array:
        """Computes the forward pass of the node."""
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    def backward(self) -> np.array:
        """Computes the backward pass of the node."""
        raise NotImplementedError("Backward method must be implemented in subclasses.")

    def vjp_fun(self, *args, **kwargs) -> np.array:
        """Computes the vector-Jacobian product for the node."""
        raise NotImplementedError("VJP function must be implemented in subclasses.")

class Value(Node):
    """This computation graph node stores a value without inputs."""

    def __init__(self, name: str, data: np.array) -> None:
        super().__init__(name)
        self.out = data

    def forward(self) -> np.array:
        self.d_out = np.zeros_like(self.out)
        return self.out

    def backward(self) -> np.array:
        return self.d_out

class Dot(Node):
    """This node computes a np.dot operation on the input nodes."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, lhs: Node, rhs: Node) -> np.array:
        self.inputs = [lhs, rhs]  # store references to input nodes
        self.out = np.dot(lhs.out, rhs.out)
        self.d_out = np.zeros_like(self.out)
        return self.out
    
    def vjp_fun_lhs(self, input: np.array, mult: np.array, vec: np.array) -> np.array:
        # print(f"input.shape, mult.shape, vec.shape = {input.shape}, {mult.shape}, {vec.shape}")
        vjp = np.dot(vec, mult.T)
        # print(f"vjp.shape = {vjp.shape}")
        return vjp

    def vjp_fun_rhs(self, input: np.array, mult: np.array, vec: np.array) -> np.array:
        # print(f"input.shape, mult.shape, vec.shape = {input.shape}, {mult.shape}, {vec.shape}")
        # vjp = np.dot(vec, mult).T
        vjp = np.dot(vec.T, mult).T
        # print(f"vjp.shape = {vjp.shape}")
        return vjp

    def backward(self) -> np.array:
        # Accumulate gradient into inputs w/ chain rule
        lhs, rhs = self.inputs[0], self.inputs[1]
        # print(f"lhs.d_out.shape: {lhs.d_out.shape}")
        # print(f"lhs.d_out: {lhs.d_out}")
        d_lhs = self.vjp_fun_lhs(lhs.out, rhs.out, self.d_out)
        # print(f"d_lhs.shape: {d_lhs.shape}")
        lhs.d_out += d_lhs

        # print(f"rhs.d_out.shape: {rhs.d_out.shape}")
        # print(f"rhs.d_out: {rhs.d_out}")
        d_rhs = self.vjp_fun_rhs(rhs.out, lhs.out, self.d_out)
        # print(f"d_rhs.shape: {d_rhs.shape}")
        rhs.d_out += d_rhs
        return self.d_out
    
class Tanh(Node):
    """This node computes a np.tanh operation on the input node."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, input: Node) -> np.array:
        self.inputs = [input]  # store references to input nodes
        self.out = np.tanh(input.out)
        self.d_out = np.zeros_like(self.out)
        return self.out
    
    def vjp_fun(self, input: np.array, vec: np.array) -> np.array:
        # The VJP w.r.t. input of np.tanh(input)
        sech2 = 1./ (np.cosh(input)**2.)
        return vec * sech2

    def backward(self) -> np.array:
        # Accumulate gradient into inputs w/ chain rule
        input = self.inputs[0]
        d_input = self.vjp_fun(input.out, self.d_out)
        input.d_out += d_input
        return self.d_out
    
class ReLU(Node):
    """This node computes a ReLU operation on the input node."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, input: Node) -> np.array:
        self.inputs = [input]  # store references to input nodes
        self.out = input.out * (input.out > 0)
        self.d_out = np.zeros_like(self.out)
        return self.out
    
    def vjp_fun(self, input: np.array, vec: np.array) -> np.array:
        # The VJP w.r.t. input of ReLU(input)
        fac = 1. * (input > 0)
        return vec * fac

    def backward(self) -> np.array:
        # Accumulate gradient into inputs w/ chain rule
        input = self.inputs[0]
        d_input = self.vjp_fun(input.out, self.d_out)
        input.d_out += d_input
        return self.d_out
