from typing import List
import numpy as np


def softmax(z: np.array, axis: int = -1) -> np.array:
    """
    Computes the softmax function safely, avoiding overflow by first
    subtracting the maximum logit value.
    See https://jaykmody.com/blog/stable-softmax/ for explanation.

    Args:
      z: An input numpy array of logits.
      axis: The axis over which we sum to normalize the outputs.

    Returns:
      A numpy array of the softmax outputs.
    """
    z = z - np.max(z, axis=axis, keepdims=True)
    num = np.exp(z)
    den = np.sum(num, axis=axis, keepdims=True)
    return num / den


def log_softmax(z: np.array, axis: int = -1) -> np.array:
    """
    Computes the log-softmax function safely, avoiding overflow by first
    subtracting the maximum logit value.
    See https://jaykmody.com/blog/stable-softmax/ for explanation.

    Args:
      z: An input numpy array of logits.
      axis: The axis over which we sum.

    Returns:
      A numpy array of the log-softmax outputs.
    """
    z_max = np.max(z, axis=axis, keepdims=True)
    return z - z_max - np.log(np.sum(np.exp(z - z_max), axis=axis, keepdims=True))


def cross_entropy(y_pred: np.array, true_idx: List[int]) -> np.array:
    """
    Computes the cross-entropy function safely, avoiding overflow by
    calling the safe log_softmax function.
    See https://jaykmody.com/blog/stable-softmax/ for explanation.

    Args:
      y_pred: An input numpy array of probits.
      true_idx: The index label of the true class.

    Returns:
      A numpy array of the cross-entropy.
    """
    n = y_pred.shape[0]
    return -log_softmax(y_pred)[np.arange(n), true_idx][:, np.newaxis]


def label_to_one_hot(label: int = 0, n_classes: int = 10) -> np.array:
    """
    Generates a one-hot encoded vector for a given label.

    Args:
      label: The label to encode (integer).
      n_classes: The total number of classes.

    Returns:
      A numpy array representing the one-hot encoded vector.
    """
    one_hot = np.zeros(n_classes)
    one_hot[label] = 1
    return one_hot
