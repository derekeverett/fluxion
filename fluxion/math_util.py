from typing import List, Optional
import numpy as np

# see https://jaykmody.com/blog/stable-softmax/

def softmax(z: np.array, axis: int = -1) -> np.array:
    z = z - np.max(z, axis=axis, keepdims=True)
    num = np.exp(z)
    den = np.sum(num, axis=axis, keepdims=True)
    return num / den

def log_softmax(z: np.array, axis: int = -1):
    z_max = np.max(z, axis=axis, keepdims=True)
    return z - z_max - np.log(np.sum(np.exp(z - z_max), axis=axis, keepdims=True))

def cross_entropy(y_pred: np.array, true_idx: List[int]):
    n = y_pred.shape[0]
    return -log_softmax(y_pred)[np.arange(n), true_idx][:, np.newaxis]

def label_to_one_hot(label: int=0, n_classes: int=10):
  """
  Generates a one-hot encoded vector for a given label.

  Args:
    label: The label to encode (integer).
    n_classes: The total number of classes.

  Returns:
    A NumPy array representing the one-hot encoded vector.
  """
  one_hot = np.zeros(n_classes)
  one_hot[label] = 1
  return one_hot
