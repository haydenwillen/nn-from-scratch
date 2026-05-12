"""Activation functions and their derivatives.

Each activation is paired with its derivative so backpropagation can
look them up by name. Derivatives are written in terms of the
pre-activation z, matching the form used in network.py.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Split on sign to avoid overflow in exp for large |z|.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS: dict[str, tuple] = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
}


def get(name: str) -> tuple:
    """Look up (activation, derivative) by name."""
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {name!r}. Available: {list(ACTIVATIONS)}"
        )
    return ACTIVATIONS[name]