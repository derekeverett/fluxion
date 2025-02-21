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


class Adam(Optimizer):
    """Adam optimization algorithm from https://arxiv.org/pdf/1412.6980."""

    def __init__(
        self, name: str, lr: float, values: np.array, beta1: float, beta2: float
    ) -> None:
        super().__init__(name, lr, values)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = beta1
        self.beta2_t = beta2
        self.mom_m = np.zeros_like(values)
        self.mom_v = np.zeros_like(values)

    def step(self, grad: np.array) -> None:
        """Steps Adam one iteration.

        Args:
            grad: a numpy array containing the gradient w.r.t. values."""

        # m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
        self.mom_m = self.beta1 * self.mom_m + (1.0 - self.beta1) * grad
        # v(t) = beta2 * v(t-1) + (1 – beta2) * g(t)^2
        self.mom_v = self.beta2 * self.mom_v + (1.0 - self.beta2) * grad**2.0
        # bias correction
        m_hat = self.mom_m / (1.0 - self.beta1_t)
        v_hat = self.mom_v / (1.0 - self.beta2_t)
        # x(t) = x(t-1) – alpha * mhat(t) / (sqrt(vhat(t)) + eps)
        self.values = self.values - self.lr * (m_hat / np.sqrt(v_hat + 1e-5))
        # update beta1(t) and beta2(t)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
