# nn — a neural network from scratch, in NumPy

A two-layer feedforward neural network for regression, built without any ML framework. The goal is clarity: every line of math is in plain NumPy and the whole network fits in five files.

## Why

To make sure I actually understand what a neural network is doing — forward pass, backprop, gradient descent — rather than just calling `model.fit()`.

## Layout

```
nn/
  activations.py   # sigmoid, relu, tanh + derivatives
  data.py          # synthetic data, train/val/test split, standardization
  network.py       # init_params, forward, backward, predict
  training.py      # mse loss, SGD step, training loop
tests/
  test_gradients.py  # numerical gradient checks vs analytical backprop
  test_training.py   # end-to-end smoke tests
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```python
from nn.data import make_synthetic_regression, split, standardize
from nn.training import train, mse
from nn.network import predict

X, Y = make_synthetic_regression(n_samples=1000, seed=0)
s = split(X, Y, seed=0)
X_train, X_val, X_test = standardize(s.X_train, s.X_val, s.X_test)

params, hist = train(
    X_train, s.Y_train,
    X_val=X_val, Y_val=s.Y_val,
    hidden_dim=32, lr=1e-2, epochs=2000, seed=0,
)
print("test_mse =", mse(s.Y_test, predict(X_test, params)))
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests verify (a) analytical gradients from `backward()` match finite differences to ~1e-6 across all activations, and (b) the training loop fits the synthetic data and generalizes to held-out samples.

## What's intentionally missing

- No PyTorch / TensorFlow / sklearn — the point is to write the math.
- No mini-batching — full-batch GD keeps the loop readable. Easy extension.
- No regularization, no momentum, no Adam — also easy extensions.
- Linear output layer only — classification (softmax + cross-entropy) is a natural next step.

## Notes

- All gradients are averaged over the batch in `backward`.
- `backward()` computes the gradient of (1/2) · MSE; the factor of 2 is absorbed into the learning rate (this is why `lr` and reported `train_loss` use slightly different scalings).
- Weight init uses small Gaussian noise scaled by `init_scale`; biases are zero. For ReLU specifically, He init would be more principled.
- Standardization statistics come from the training set only (no test leakage).