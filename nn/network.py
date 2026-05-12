"""Two-layer feedforward neural network, NumPy only.

Conventions
-----------
- X has shape (n_samples, input_dim).
- Y has shape (n_samples, output_dim).
- W1 has shape (input_dim, hidden_dim); b1 has shape (1, hidden_dim).
- W2 has shape (hidden_dim, output_dim); b2 has shape (1, output_dim).
- All gradients are averaged over the batch (divide by n_samples).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import activations


@dataclass
class Params:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class Cache:
    """Intermediate values from the forward pass, needed for backprop."""
    X: np.ndarray
    Z1: np.ndarray
    A1: np.ndarray
    Z2: np.ndarray
    Y_hat: np.ndarray


def init_params(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    scale: float = 0.01,
    seed: int | None = None,
) -> Params:
    """Initialize weights with small Gaussian noise; biases at zero."""
    rng = np.random.default_rng(seed)
    return Params(
        W1=rng.standard_normal((input_dim, hidden_dim)) * scale,
        b1=np.zeros((1, hidden_dim)),
        W2=rng.standard_normal((hidden_dim, output_dim)) * scale,
        b2=np.zeros((1, output_dim)),
    )


def forward(
    X: np.ndarray,
    params: Params,
    activation: str = "relu",
    output: str = "linear",
) -> Cache:
    """Forward pass.

    output: 'linear' for regression, 'softmax' for classification.
    """
    act_fn, _ = activations.get(activation)
    Z1 = X @ params.W1 + params.b1
    A1 = act_fn(Z1)
    Z2 = A1 @ params.W2 + params.b2
    if output == "linear":
        Y_hat = Z2
    elif output == "softmax":
        Y_hat = activations.softmax(Z2)
    else:
        raise ValueError(f"Unknown output mode {output!r}")
    return Cache(X=X, Z1=Z1, A1=A1, Z2=Z2, Y_hat=Y_hat)


def backward(
    Y: np.ndarray,
    cache: Cache,
    params: Params,
    activation: str = "relu",
    output: str = "linear",
) -> Params:
    """Backprop. For linear output: MSE gradient. For softmax: cross-entropy."""
    _, act_deriv = activations.get(activation)
    n = Y.shape[0]

    # Both losses give dZ2 = (Y_hat - Y) / n at the output layer.
    # For MSE + linear: this is the standard residual.
    # For cross-entropy + softmax: the softmax Jacobian and the
    # cross-entropy gradient combine to give exactly the same form.
    # Choice of output mode doesn't change this line.
    dZ2 = (cache.Y_hat - Y) / n
    dW2 = cache.A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)

    dA1 = dZ2 @ params.W2.T
    dZ1 = dA1 * act_deriv(cache.Z1)
    dW1 = cache.X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)

    return Params(W1=dW1, b1=db1, W2=dW2, b2=db2)


def predict(
    X: np.ndarray,
    params: Params,
    activation: str = "relu",
    output: str = "linear",
) -> np.ndarray:
    return forward(X, params, activation, output).Y_hat