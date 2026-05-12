# nn — a neural network from scratch, in NumPy

A two-layer neural network for regression. No PyTorch, no autograd, no `model.fit()` — just the math, written out in NumPy.

I built this to make sure I actually understood what a neural network does under the hood. Forward pass, backprop, gradient descent: it's easy to call a library and watch the loss go down. Writing it yourself is how you find out where the gaps in your understanding actually are.

The whole thing fits in five files:

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

## Running it

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

Two things get tested: the analytical gradients from `backward()` match central finite differences to ~1e-6 across all three activations, and the training loop actually fits the synthetic data and generalizes to held-out samples. The first catches math bugs, the second catches everything else.

## What's not here

No PyTorch, TensorFlow, or sklearn — that's the whole point.

No mini-batching either. Full-batch gradient descent keeps the loop short enough to read in one sitting, which matters more than convergence speed on a 1000-sample toy problem. Same reasoning for skipping momentum, Adam, and any form of regularization.

The output layer is linear, so this only does regression. Softmax + cross-entropy is the natural next thing to add.

## A few things worth knowing

`backward()` differentiates ½·MSE, not MSE. The factor of 2 gets absorbed into the learning rate — standard convention, but worth flagging since `train_loss` in the history reports MSE and the gradient is technically of something else. The gradient check in `test_gradients.py` compares against ½·MSE to keep them consistent.

Weights are initialized with small Gaussian noise scaled by `init_scale`; biases start at zero. He init would be more principled for ReLU. Haven't done it yet.

Standardization uses training-set statistics only, applied to val and test. Easy thing to get wrong if you're not paying attention.