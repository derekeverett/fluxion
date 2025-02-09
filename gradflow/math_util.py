from typing import List, Optional
import numpy as np

# see https://jaykmody.com/blog/stable-softmax/


def softmax(z: np.array, axis: int = -1):
    z = z - np.max(z, axis=axis, keep_dim=True)
    num = np.exp(z)
    den = np.sum(num, axis=axis, keep_dim=True)


def log_softmax(z: np.array, axis: int = -1):
    z_max = np.max(z, axis=axis, keep_dim=True)
    return z - z_max - np.log(np.sum(np.exp(z - z_max), axis=axis))


def cross_entropy(y_pred: np.array, true_idx: List[int]):
    n = y_pred.shape[0]
    return -log_softmax(y_pred)[np.arange(n), true_idx]
