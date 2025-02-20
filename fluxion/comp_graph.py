from typing import List, Optional
import numpy as np
from fluxion.math_util import softmax, cross_entropy, label_to_one_hot
from fluxion.optimize import Optimizer


class Node:
    """
    A base class for computation graph nodes.
    Inspired by https://github.com/davidrosenberg/mlcourse/blob/gh-pages/Notebooks/computation-graph/computation-graph-framework.ipynb.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.out: Optional[np.array] = None
        self.d_out: Optional[np.array] = None
        self.inputs: List["Node"] = []

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

    def __init__(
        self,
        name: str,
        data: np.array,
        optimizer: Optimizer = None,
        optimize: bool = False,
    ) -> None:
        super().__init__(name)
        self.out = data
        self.optimize = optimize
        self.optimizer = optimizer

    def forward(self) -> np.array:
        """
        Computes the forward pass.

        Returns:
            A numpy array of the value currently stored.
        """
        self.d_out = np.zeros_like(self.out)
        return self.out

    def backward(self) -> None:
        """
        Computes the backward pass.
        There are no parent nodes to accumulate gradients into.
        """

    def update(self, new_data: np.array) -> None:
        """
        Updates the values stored by the node.

        Args:
            new_data: A numpy array defining the updated values.
        """
        self.out = new_data


class Dot(Node):
    """This node computes a np.dot operation on the input nodes."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, lhs: Node, rhs: Node) -> np.array:
        """
        Computes np.dot(lhs, rhs) operation on the inputs.

        Args:
            lhs: A numpy array of the left argument to np.dot.
            rhs: A numpy array of the right argument to np.dot.

        Returns:
            A numpy array of the ouput.
        """
        self.inputs = [lhs, rhs]
        self.out = np.dot(lhs.out, rhs.out)
        self.d_out = np.zeros_like(self.out)
        return self.out

    def vjp_fun_lhs(self, input: np.array, mult: np.array, vec: np.array) -> np.array:
        """
        Computes the vector-Jacobian product.

        Args:
            input: A numpy array of the variables for which the Jacobian of Dot(input, mult) is w.r.t.
            mult: A numpy array of the quantity multipling input in the Dot(input, mult) function.
            vec: A numpy array of the vector multiplying the Jacobian.

        Returns:
            A numpy array of the vector-Jacobian product.
        """
        vjp = np.dot(vec, mult.T)
        return vjp

    def vjp_fun_rhs(self, input: np.array, mult: np.array, vec: np.array) -> np.array:
        """
        Computes the vector-Jacobian product.

        Args:
            input: A numpy array of the variables for which the Jacobian of Dot(mult, input) is w.r.t.
            mult: A numpy array of the quantity multipling input in the Dot(mult, input) function.
            vec: A numpy array of the vector multiplying the Jacobian.

        Returns:
            A numpy array of the vector-Jacobian product.
        """
        vjp = np.dot(vec.T, mult).T
        return vjp

    def backward(self) -> None:
        """
        Computes the backward pass over the node.

        """
        lhs, rhs = self.inputs[0], self.inputs[1]
        d_lhs = self.vjp_fun_lhs(lhs.out, rhs.out, self.d_out)
        lhs.d_out += d_lhs
        d_rhs = self.vjp_fun_rhs(rhs.out, lhs.out, self.d_out)
        rhs.d_out += d_rhs


class Tanh(Node):
    """This node computes a np.tanh operation on the input node."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, input: Node) -> np.array:
        """
        Computes np.tanh(input) operation on the input.

        Args:
            input: A numpy array of the input to np.tanh.

        Returns:
            A numpy array of the ouput.
        """

        self.inputs = [input]
        self.out = np.tanh(input.out)
        self.d_out = np.zeros_like(self.out)
        return self.out

    def vjp_fun(self, input: np.array, vec: np.array) -> np.array:
        """
        Computes the vector-Jacobian product.

        Args:
            input: A numpy array of the variables for which the Jacobian of np.tanh(input) is w.r.t.
            vec: A numpy array of the vector multiplying the Jacobian.
        Returns:
            A numpy array of the vector-Jacobian product.
        """

        sech2 = 1.0 / (np.cosh(input) ** 2.0)
        return vec * sech2

    def backward(self) -> None:
        input = self.inputs[0]
        d_input = self.vjp_fun(input.out, self.d_out)
        input.d_out += d_input


class ReLU(Node):
    """This node computes a ReLU operation on the input node."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, input: Node) -> np.array:
        """
        Computes ReLU(input) operation on the input.

        Args:
            input: A numpy array of the input to ReLU.

        Returns:
            A numpy array of the ouput.
        """

        self.inputs = [input]
        self.out = input.out * (input.out > 0)
        self.d_out = np.zeros_like(self.out)
        return self.out

    def vjp_fun(self, input: np.array, vec: np.array) -> np.array:
        """
        Computes the vector-Jacobian product.

        Args:
            input: A numpy array of the variables for which the Jacobian of ReLU(input) is w.r.t.
            vec: A numpy array of the vector multiplying the Jacobian.

        Returns:
            A numpy array of the vector-Jacobian product.
        """

        fac = 1.0 * (input > 0)
        return vec * fac

    def backward(self) -> None:
        """
        Computes the backward pass over the node.

        """
        input = self.inputs[0]
        d_input = self.vjp_fun(input.out, self.d_out)
        input.d_out += d_input


class MSELoss(Node):
    """This node computes the mean squared error."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def forward(self, y: Node, y_true: np.array) -> np.array:
        """
        Computes the mean squared error.

        Args:
            y: A numpy array of the predicted values.
            y_true: A numpy array of the true values.

        Returns:
            A numpy array of the ouput.
        """
        self.inputs = [y]
        err = y.out - y_true
        self.err = err
        n = err.shape[0]
        self.out = np.dot(err.T, err) / n
        self.d_out = np.zeros_like(self.out)
        return self.out

    def vjp_fun(self, vec: np.array) -> np.array:
        """
        Computes the vector-Jacobian product.

        Args:
            vec: A numpy array of the vector multiplying the Jacobian.

        Returns:
            A numpy array of the vector-Jacobian product.

        """
        n = self.err.shape[0]
        return vec * 2 * self.err / n

    def backward(self) -> None:
        input = self.inputs[0]
        d_input = self.vjp_fun(self.d_out)
        input.d_out += d_input


class CrossEntropyLoss(Node):
    """This node computes the mean cross entropy loss."""

    def __init__(self, name: str, n_classes: int) -> None:
        super().__init__(name)
        self.n_classes = n_classes

    def forward(self, y: Node, true_idxs: List[int]) -> np.array:
        """
        Computes the mean cross entropy loss.

        Args:
            y: A numpy array of the predicted values.
            true_idxs: A list of the true indices.

        Returns:
            A numpy array of the ouput.
        """

        self.inputs = [y]
        y_true = np.array(
            [
                label_to_one_hot(label=true_idx, n_classes=self.n_classes)
                for true_idx in true_idxs
            ]
        )
        self.err = softmax(y.out) - y_true
        self.out = np.mean(cross_entropy(y.out, true_idxs), keepdims=True)
        self.d_out = np.zeros_like(self.out)
        return self.out

    # see https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
    def vjp_fun(self, vec: np.array) -> np.array:
        """
        Computes the vector-Jacobian product.

        Args:
            vec: A numpy array of the vector multiplying the Jacobian.
        Returns:
            A numpy array of the vector-Jacobian product.
        """
        n = self.err.shape[0]
        return vec * self.err / n

    def backward(self) -> None:
        input = self.inputs[0]
        d_input = self.vjp_fun(self.d_out)
        input.d_out += d_input


class Graph:
    """A base class for computation graphs."""

    def __init__(self, name: str, out_nodes: List["Node"]) -> None:
        self.name = name
        self.out_nodes = out_nodes
        self.all_nodes = set()

    def topo_sort(self) -> List["Node"]:
        """
        Computes a (reverse) topological sort of the graph.

        Returns:
            A list of nodes in reverse topological order.
        """

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

        self.all_nodes = visited

        return reversed(topo_sorted)

    def backward(self) -> None:
        """
        Computes the backward pass over the graph.
        """
        # do topo sort
        topo_sorted = self.topo_sort()
        # initiate the vjp of each output node
        for node in self.out_nodes:
            node.d_out = np.array([[1.0]])
        # now call backward on every node in topo order
        for node in topo_sorted:
            node.backward()

    def step_optimizer(self):
        """
        Updates the values stored by the Value nodes in the graph with flag optimize=True.

        """
        for node in self.all_nodes:
            if isinstance(node, Value) and node.optimize:
                node.optimizer.step(node.d_out)
                node.update(node.optimizer.values)
