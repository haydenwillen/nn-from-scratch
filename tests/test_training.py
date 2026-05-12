"""Smoke tests for the training loop.

These don't verify correctness of individual gradients (test_gradients
does that). They verify that the training loop, taken end-to-end,
actually fits the data — catching bugs like wrong gradient sign,
parameters not being updated, or a broken forward pass that the
gradient check wouldn't notice because it tests backward in isolation.
"""

from __future__ import annotations

import pytest

from nn.data import make_synthetic_regression, split, standardize
from nn.training import mse, train
from nn.network import predict


@pytest.fixture
def data():
    X, Y = make_synthetic_regression(n_samples=500, seed=0)
    s = split(X, Y, seed=0)
    X_train, X_val, X_test = standardize(s.X_train, s.X_val, s.X_test)
    return X_train, s.Y_train, X_val, s.Y_val, X_test, s.Y_test


def test_train_loss_decreases(data):
    X_train, Y_train, _, _, _, _ = data
    _, history = train(
        X_train, Y_train,
        hidden_dim=16, lr=1e-2, epochs=2000, seed=0, log_every=0,
    )
    # Loss should drop by at least an order of magnitude on this problem.
    assert history.train_loss[-1] < history.train_loss[0] / 10, (
        f"train loss barely moved: {history.train_loss[0]:.4f} -> "
        f"{history.train_loss[-1]:.4f}"
    )


def test_generalizes_to_held_out_data(data):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data
    params, _ = train(
        X_train, Y_train,
        X_val=X_val, Y_val=Y_val,
        hidden_dim=16, lr=1e-2, epochs=2000, seed=0, log_every=0,
    )
    test_mse = mse(Y_test, predict(X_test, params))
    train_mse = mse(Y_train, predict(X_train, params))
    # Test error should be in the same ballpark as train — not 10x worse.
    assert test_mse < train_mse * 3, (
        f"test_mse={test_mse:.4f} much worse than train_mse={train_mse:.4f}"
    )
    # And test_mse should be meaningfully below the variance of Y
    # (which is roughly what you'd get by predicting the mean).
    baseline = float(Y_train.var())
    assert test_mse < baseline * 0.5, (
        f"test_mse={test_mse:.4f} not much better than mean baseline "
        f"{baseline:.4f}"
    )
