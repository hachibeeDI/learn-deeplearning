import numpy as np


def step_function(x: np.ndarray) -> np.ndarray:
    return np.array(x > 0, dtype=np.int)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity(x):
    return x


def _softmax_could_overflow(a):
    exp_a = np.exp(a)
    y = exp_a / np.sum(exp_a)
    return y


def softmax(a):
    """
    The summed value is 1.0 whatever argument is.
    That means we can use each values as probability.

    >>> a = np.array([0.3, 2.9, 4.0])
    >>> y = softmax(a)

    >>> np.sum(y)
    1.0
    """
    c = np.max(a)

    exp_a = np.exp(a - c)
    y = exp_a / np.sum(exp_a)
    return y
