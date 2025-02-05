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
        self.inputs: List['Node'] = []

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

    def __init__(self, name: str, data: np.array, optimize: bool = False) -> None:
        super().__init__(name)
        self.out = data
        self.optimize = optimize

    def forward(self) -> np.array:
        self.d_out = np.zeros_like(self.out)
        return self.out

    def backward(self) -> np.array:
        return self.d_out

    def update(self, new_data: np.array):
        self.out = new_data

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
    
class MSELoss(Node):
    """This node computes the mean squared error."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, y: Node, y_true: np.array) -> np.array:
        self.inputs = [y]  # store references to input nodes
        err = (y.out - y_true)
        self.err = err
        n = err.shape[0]
        # y.out and y_true should be (B, 1) where B is batch size
        # so the mean squared error is np.dot(err.T, err) / n
        self.out = np.dot(err.T, err) / n
        self.d_out = np.zeros_like(self.out)
        return self.out
    
    def vjp_fun(self, vec: np.array) -> np.array:
        # The VJP f(y) = (err)^2 = (y - y_true)^2 w.r.t. y
        n = self.err.shape[0]
        # return np.dot(vec, 2*self.err/n)
        return vec * 2*self.err/n

    def backward(self) -> np.array:
        # Accumulate gradient into inputs w/ chain rule
        input = self.inputs[0]
        d_input = self.vjp_fun(self.d_out)
        input.d_out += d_input
        return self.d_out

class Graph:
    """A base class for computation graphs."""
    
    def __init__(self, name: str, out_nodes: List['Node']) -> None:
        self.name = name
        self.out_nodes = out_nodes
        self.all_nodes = set()

    def topo_sort(self) -> List['Node']:
        topo_sorted = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for in_node in v.inputs:
                    build_topo(in_node)
                topo_sorted.append(v)
        
        for node in self.out_nodes:
            build_topo(node)

        # store all visited nodes in self.all_nodes
        self.all_nodes = visited

        return reversed(topo_sorted)

    def backward(self) -> None:
        """Computes the backward pass over the graph."""
        # do topo sort
        topo_sorted = self.topo_sort()
        # initiate the vjp of each output node
        for node in self.out_nodes:
            node.d_out = np.array([[1.]])
        # now call backward on every node in topo order
        for node in topo_sorted:
            node.backward()

    def step_optimizer(self, **kwargs):
        """Step optimizer for all Value nodes in a Graph with flag optimize=True."""
        for node in self.all_nodes:
            if node is Value and node.optimize:
                # Gradient Descent implemented inline for now
                # TODO: generalize this to allow other options/better encapsulation
                lr = 1e-3
                new_data = node.out - lr * node.d_out
                node.update(new_data)