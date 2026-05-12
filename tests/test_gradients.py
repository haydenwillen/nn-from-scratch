"""Numerical gradient check for backward().

For each parameter element θ_i, compare the analytical gradient
∂L/∂θ_i from backward() against the central finite difference
    (L(θ_i + ε) - L(θ_i - ε)) / (2ε)
Relative error should be ~1e-7 or better in float64.
"""

from __future__ import annotations

import numpy as np
import pytest

from nn.data import make_synthetic_regression, standardize
from nn.network import Params, backward, forward, init_params
from nn.training import mse


def _loss(X, Y, params, activation):
    return 0.5 * mse(Y, forward(X, params, activation).Y_hat)


def _relative_error(a: float, b: float) -> float:
    denom = max(abs(a) + abs(b), 1e-12)
    return abs(a - b) / denom


def check_gradients(
    activation: str = "tanh",
    n_samples: int = 8,
    input_dim: int = 4,
    hidden_dim: int = 4,
    output_dim: int = 1,
    eps: float = 1e-6,
    tol: float = 1e-6,
    seed: int = 0,
) -> None:
    # Small problem, standardized inputs — keeps activations in the
    # well-conditioned regime where finite differences are accurate.
    X, Y = make_synthetic_regression(n_samples=n_samples, n_features=input_dim, seed=seed)
    (X,) = standardize(X)
    params = init_params(input_dim, hidden_dim, output_dim, scale=0.5, seed=seed)

    # Note on MSE scaling: backward() divides dZ2 by n but the loss
    # has no factor of 2, so analytical grads are (1/n) * ∂||r||²/∂θ
    # while finite differences of mse give (1/n) * ∂||r||²/∂θ as well.
    # They match without rescaling.
    cache = forward(X, params, activation)
    grads = backward(Y, cache, params, activation)

    for name in ("W1", "b1", "W2", "b2"):
        theta = getattr(params, name)
        analytic = getattr(grads, name)
        numeric = np.zeros_like(theta)

        it = np.nditer(theta, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            i = it.multi_index
            original = theta[i]

            theta[i] = original + eps
            loss_plus = _loss(X, Y, params, activation)
            theta[i] = original - eps
            loss_minus = _loss(X, Y, params, activation)
            theta[i] = original  # restore

            numeric[i] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()

        max_err = max(
            _relative_error(float(a), float(n))
            for a, n in zip(analytic.ravel(), numeric.ravel())
        )
        assert max_err < tol, f"{name}: max relative error {max_err:.2e} exceeds {tol:.0e}"
        print(f"{name}: max rel err {max_err:.2e}  OK")


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu"])
def test_gradients(activation):
    check_gradients(activation=activation)