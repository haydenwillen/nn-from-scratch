"""End-to-end tests for the classification mode."""

from __future__ import annotations

import numpy as np

from nn.data import make_synthetic_classification, split, standardize
from nn.network import predict
from nn.training import train


def test_classification_accuracy():
    X, Y = make_synthetic_classification(n_samples=600, n_classes=3, seed=0)
    s = split(X, Y, seed=0)
    X_train, X_val, X_test = standardize(s.X_train, s.X_val, s.X_test)
    params, _ = train(
        X_train, s.Y_train,
        X_val=X_val, Y_val=s.Y_val,
        hidden_dim=16, lr=1e-1, epochs=500, loss="cross_entropy",
        seed=0, log_every=0,
    )
    probs = predict(X_test, params, output="softmax")
    pred = probs.argmax(axis=1)
    true = s.Y_test.argmax(axis=1)
    accuracy = (pred == true).mean()
    # Well-separated clusters should give very high accuracy.
    assert accuracy > 0.9, f"accuracy={accuracy:.3f} too low"
