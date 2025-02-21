import numpy as np


class Optimizer:
    """
    A base class for optimization algorithms.
    """

    def __init__(self, name: str, lr: float, values: np.array) -> None:
        self.name = name
        self.lr = lr
        self.values = values

    def step(self, *args, **kwargs) -> np.array:
        """Steps the optimizer one iteration."""
        raise NotImplementedError("Step method must be implemented in subclasses.")


class SGD(Optimizer):
    """Stochastic Gradient Descent optimization algorithm with momentum."""

    def __init__(self, name: str, lr: float, values: np.array, beta: float) -> None:
        super().__init__(name, lr, values)
        self.beta = beta
        self.mom_vec = np.zeros_like(values)

    def step(self, grad: np.array) -> None:
        """Steps stochastic gradient descent (with momentum) one iteration.

        Args:
            grad: a numpy array containing the gradient w.r.t. values."""

        delta = self.lr * grad + self.beta * self.mom_vec
        self.mom_vec = delta
        self.values = self.values - delta
