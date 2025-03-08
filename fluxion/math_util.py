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
    one_hot[label] = 1.0
    return one_hot


def get_im2col_indices(img_shape, field_height, field_width, stride=1, pad=0):
    """
    A function to find the index order for producing the im2col matrix.

    Args:
        img_shape: The shape of the input image numpy array.
        field_height: The height of the filter receptive field.
        field_width: The width of the filter receptive field.
        stride: The filter stride.
        pad: The amount of zeroes to pad to the height and width of image.

    Returns:
        A tuple containg the indices for the channel, height and width.

    """
    batch_size, chan_in, height_in, width_in = img_shape
    out_height = (height_in + 2 * pad - field_height) // stride + 1
    out_width = (width_in + 2 * pad - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, chan_in)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * chan_in)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    height_indices = i0.reshape(-1, 1) + i1.reshape(1, -1)
    width_indices = j0.reshape(-1, 1) + j1.reshape(1, -1)

    chan_indices = np.repeat(np.arange(chan_in), field_height * field_width).reshape(
        -1, 1
    )

    return (chan_indices, height_indices, width_indices)


def im2col(img, field_height, field_width, stride=1, pad=0):
    """
    An implementation of im2col based on some fancy indexing.

    Args:
        img: The input image numpy array.
        field_height: The height of the filter receptive field.
        field_width: The width of the filter receptive field.
        stride: The filter stride.
        pad: The amount of zeroes to pad to the height and width of image.

    Returns:
        A matrix with receptive field patches flattened and appended to a columns.

    """
    # Zero-pad the input
    x_padded = np.pad(img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    (chan_indices, height_indices, width_indices) = get_im2col_indices(
        img.shape, field_height, field_width, stride=stride, pad=pad
    )
    cols = x_padded[:, chan_indices, height_indices, width_indices]
    chan_in = img.shape[1]
    col_mat = cols.transpose(1, 2, 0).reshape(field_height * field_width * chan_in, -1)
    return col_mat


def col2im(col_mat, img_shape, field_height=3, field_width=3, stride=1, pad=1):
    """
    An implementation of col2im based on some fancy indexing.

    Args:
        col_mat: The matrix of the input images receptive field patches flattened.
        img_shape: The shape of the input image numpy array.
        field_height: The height of the receptive field.
        field_width: The width of the receptive field.
        stride: The filter stride.
        pad: The amount of zeroes to pad to the height and width of image.

    Returns:
        A numpy array of the padded image of shape (batch_size, channels, height, width).

    """
    batch_size, chan_in, height_in, width_in = img_shape
    height_padded, width_padded = height_in + 2 * pad, width_in + 2 * pad
    img_padded = np.zeros(
        (batch_size, chan_in, height_padded, width_padded), dtype=col_mat.dtype
    )
    chan_indices, height_indices, width_indices = get_im2col_indices(
        img_shape, field_height, field_width, stride, pad
    )
    cols_reshaped = col_mat.reshape(
        chan_in * field_height * field_width, -1, batch_size
    )
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(
        img_padded,
        (slice(None), chan_indices, height_indices, width_indices),
        cols_reshaped,
    )
    if pad == 0:
        return img_padded
    return img_padded[:, :, pad:-pad, pad:-pad]
