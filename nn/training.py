"""Training loop, loss function, and parameter updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import Params, backward, forward, init_params


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def cross_entropy(Y: np.ndarray, Y_hat: np.ndarray, eps: float = 1e-12) -> float:
    """Categorical cross-entropy. Y is one-hot, Y_hat is softmax output."""
    # Clip to avoid log(0).
    Y_hat = np.clip(Y_hat, eps, 1.0 - eps)
    return float(-np.sum(Y * np.log(Y_hat)) / Y.shape[0])

def sgd_step(params: Params, grads: Params, lr: float) -> Params:
    """One in-place gradient descent step.

    Operates in place for clarity; returns the same Params for chaining.
    """
    params.W1 -= lr * grads.W1
    params.b1 -= lr * grads.b1
    params.W2 -= lr * grads.W2
    params.b2 -= lr * grads.b2
    return params


@dataclass
class History:
    train_loss: list[float]
    val_loss: list[float]


def train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    hidden_dim: int,
    *,
    X_val: np.ndarray | None = None,
    Y_val: np.ndarray | None = None,
    activation: str = "relu",
    loss: str = "mse",
    lr: float = 1e-2,
    epochs: int = 1000,
    init_scale: float = 0.01,
    seed: int | None = 0,
    log_every: int = 100,
) -> tuple[Params, History]:
    """Train a two-layer network with full-batch gradient descent.

    loss: 'mse' (linear output) or 'cross_entropy' (softmax output).
    """
    if loss == "mse":
        output = "linear"
        loss_fn = mse
    elif loss == "cross_entropy":
        output = "softmax"
        loss_fn = cross_entropy
    else:
        raise ValueError(f"Unknown loss {loss!r}")

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1] if Y_train.ndim > 1 else 1
    Y_train = Y_train.reshape(-1, output_dim)
    if Y_val is not None:
        Y_val = Y_val.reshape(-1, output_dim)

    params = init_params(input_dim, hidden_dim, output_dim, init_scale, seed)
    history = History(train_loss=[], val_loss=[])

    for epoch in range(epochs):
        cache = forward(X_train, params, activation, output)
        train_loss = loss_fn(Y_train, cache.Y_hat)
        history.train_loss.append(train_loss)

        if X_val is not None and Y_val is not None:
            val_pred = forward(X_val, params, activation, output).Y_hat
            history.val_loss.append(loss_fn(Y_val, val_pred))

        grads = backward(Y_train, cache, params, activation, output)
        sgd_step(params, grads, lr)

        if log_every and epoch % log_every == 0:
            msg = f"epoch {epoch:5d}  train_loss={train_loss:.6f}"
            if history.val_loss:
                msg += f"  val_loss={history.val_loss[-1]:.6f}"
            print(msg)

    return params, history