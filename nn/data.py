"""A nonlinear regression target so a NN actually beats linear regression.

The target combines a squared term, a sin term, and an interaction
between the first and last features. Works for any n_features >= 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray


def make_synthetic_regression(
    n_samples: int = 1000,
    n_features: int = 4,
    noise: float = 0.1,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A nonlinear regression target so a NN actually beats linear regression."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # Nonlinear target: sum of squares + interaction + sin.
    y = (
        (X[:, 0] ** 2)
        + np.sin(X[:, 1] if n_features > 1 else X[:, 0])
        + X[:, 0] * X[:, -1]
        + noise * rng.standard_normal(n_samples)
    )
    return X, y.reshape(-1, 1)


def split(
    X: np.ndarray,
    Y: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int | None = 0,
) -> Split:
    """Shuffle and split into train/val/test."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return Split(
        X_train=X[train_idx], Y_train=Y[train_idx],
        X_val=X[val_idx], Y_val=Y[val_idx],
        X_test=X[test_idx], Y_test=Y[test_idx],
    )


def standardize(
    X_train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Zero-mean, unit-variance using train statistics only."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return tuple((arr - mean) / std for arr in (X_train, *others))

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer class labels to one-hot encoding."""
    y = y.astype(int).ravel()
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def make_synthetic_classification(
    n_samples: int = 1000,
    n_features: int = 4,
    n_classes: int = 3,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic classification problem with linearly-separable cluster centers.

    Returns X of shape (n_samples, n_features) and Y of shape
    (n_samples, n_classes) as one-hot labels.
    """
    rng = np.random.default_rng(seed)
    # Random cluster centers, well-separated.
    centers = rng.standard_normal((n_classes, n_features)) * 3
    labels = rng.integers(0, n_classes, size=n_samples)
    X = centers[labels] + rng.standard_normal((n_samples, n_features))
    return X, one_hot(labels, n_classes)