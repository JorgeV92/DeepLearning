import argparse

import numpy as np

from backpropagation import (
    Value,
    affine_backward,
    affine_forward,
    relu_backward,
    relu_forward,
    softmax_cross_entropy,
)
from optimization import eval_numerical_gradient


class MLP:
    def __init__(self, input_dim, hidden_dims=(64,), num_classes=10,
                 seed=42, dtype=np.float32):
        self.layer_sizes = (input_dim, *hidden_dims, num_classes)
        if any(not isinstance(size, (int, np.integer)) or size <= 0
               for size in self.layer_sizes):
            raise ValueError("all layer sizes must be positive integers")
        if np.dtype(dtype).kind != "f":
            raise TypeError("parameters must use a floating-point dtype")
        self.num_layers = len(self.layer_sizes) - 1
        self.params = {}
        rng = np.random.default_rng(seed)
        for layer, (fan_in, fan_out) in enumerate(
            zip(self.layer_sizes[:-1], self.layer_sizes[1:]), start=1
        ):
            scale = np.sqrt((2 if layer < self.num_layers else 1) / fan_in)
            self.params[f"W{layer}"] = (
                rng.standard_normal((fan_in, fan_out)) * scale
            ).astype(dtype)
            self.params[f"b{layer}"] = np.zeros(fan_out, dtype=dtype)

    @property
    def num_parameters(self):
        return sum(parameter.size for parameter in self.params.values())

    def _validate_X(self, X):
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"X must be a nonempty (N, {self.layer_sizes[0]}) array")

    def _validate_y(self, X, y):
        if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must be an integer array with shape (N,)")
        if np.any(y < 0) or np.any(y >= self.layer_sizes[-1]):
            raise ValueError("labels must be between 0 and num_classes - 1")

    def forward(self, X):
        self._validate_X(X)
        activations = X
        caches = []
        for layer in range(1, self.num_layers + 1):
            activations, affine_cache = affine_forward(
                activations, self.params[f"W{layer}"], self.params[f"b{layer}"]
            )
            relu_cache = None
            if layer < self.num_layers:
                activations, relu_cache = relu_forward(activations)
            caches.append((affine_cache, relu_cache))
        return activations, caches

    def loss(self, X, y, reg=1e-4):
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg must be finite and nonnegative")
        self._validate_X(X)
        self._validate_y(X, y)
        scores, caches = self.forward(X)
        loss, upstream = softmax_cross_entropy(scores, y)
        gradients = {}
        for layer in range(self.num_layers, 0, -1):
            affine_cache, relu_cache = caches[layer - 1]
            if relu_cache is not None:
                upstream = relu_backward(upstream, relu_cache)
            upstream, dW, db = affine_backward(upstream, affine_cache)
            W = self.params[f"W{layer}"]
            loss += reg * np.sum(W * W)
            gradients[f"W{layer}"] = dW + 2 * reg * W
            gradients[f"b{layer}"] = db
        return float(loss), gradients

    def train(self, X, y, learning_rate=0.05, reg=1e-4, num_iters=200,
              batch_size=128, seed=42):
        self._validate_X(X)
        self._validate_y(X, y)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(reg) or reg < 0:
            raise ValueError("reg must be finite and nonnegative")
        for name, value in (("num_iters", num_iters), ("batch_size", batch_size)):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(num_iters):
            indices = rng.choice(len(X), min(batch_size, len(X)), replace=False)
            loss, gradients = self.loss(X[indices], y[indices], reg)
            history.append(loss)
            # Same SGD rule as optimization.py, now for every layer's W and b.
            for name in self.params:
                self.params[name] -= learning_rate * gradients[name]
        return history

    def pred(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)

    def accuracy(self, X, y):
        self._validate_X(X)
        self._validate_y(X, y)
        return float(np.mean(self.pred(X) == y))


def run_neuron_example():
    """One ReLU neuron, using Value"""
    w0, w1, bias = Value(0.5), Value(-1.0), Value(0.5)
    activation = (w0 * 2.0 + w1 * 1.0 + bias).relu()
    activation.backward()
    print(f"Value neuron: ReLU(0.5*2 - 1*1 + 0.5) = {activation.data}")
    print(f"Activation gradients: dw0={w0.grad}, dw1={w1.grad}, db={bias.grad}")


def run_gradient_checks():
    """Check every layer on a tiny float64 network; avoid ReLU kinks."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4, 3))
    y = np.array([0, 1, 1, 0])
    model = MLP(3, hidden_dims=(4, 3), num_classes=2, dtype=np.float64)
    # Nonzero biases avoid exact-zero preactivations from inactive earlier layers.
    for name in model.params:
        if name.startswith("b"):
            model.params[name][:] = rng.normal(scale=0.1, size=model.params[name].shape)
    _, analytical = model.loss(X, y, reg=0.1)
    errors = {}
    for name, parameter in model.params.items():
        def objective(candidate):
            original = model.params[name]
            model.params[name] = candidate
            try:
                return model.loss(X, y, reg=0.1)[0]
            finally:
                model.params[name] = original

        numerical = eval_numerical_gradient(objective, parameter)
        errors[name] = float(np.max(np.abs(analytical[name] - numerical)))
    return errors


def run_xor_demo():
    """Learn XOR: class 1 when the two inputs differ, using -1/+1 encoding."""
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    model = MLP(2, hidden_dims=(8,), num_classes=2)
    initial_loss = model.loss(X, y, reg=0)[0]
    history = model.train(X, y, learning_rate=0.1, reg=0, num_iters=500, batch_size=4)
    final_loss = model.loss(X, y, reg=0)[0]
    print(f"XOR (2 -> 8 -> 2): loss {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"Predictions: {model.pred(X)}; labels: {y}; accuracy: {model.accuracy(X, y):.0%}")
    return {"model": model, "loss_history": history, "initial_loss": initial_loss,
            "final_loss": final_loss}


def run_cifar10_demo(num_train=1000, hidden_dims=(64,), num_iters=200,
                     batch_size=128, learning_rate=0.05, reg=1e-4):
    """Train on a CIFAR-10 subset and evaluate on the existing validation split."""
    from utils import load_cifar10

    if not isinstance(num_train, (int, np.integer)) or not 1 <= num_train <= 49000:
        raise ValueError("num_train must be an integer between 1 and 49000")
    X_train, y_train, X_val, y_val, _, _ = load_cifar10()
    indices = np.random.default_rng(42).choice(len(X_train), num_train, replace=False)
    X_train, y_train = X_train[indices], y_train[indices]
    model = MLP(X_train.shape[1], hidden_dims=hidden_dims, num_classes=10)
    initial_loss = model.loss(X_train, y_train, reg)[0]
    print(f"CIFAR-10 MLP: {' -> '.join(map(str, model.layer_sizes))}")
    print(f"{model.num_parameters:,} parameters; {num_train} images; {num_iters} updates")
    history = model.train(X_train, y_train, learning_rate=learning_rate, reg=reg,
                          num_iters=num_iters, batch_size=batch_size)
    final_loss = model.loss(X_train, y_train, reg)[0]
    train_accuracy = model.accuracy(X_train, y_train)
    val_accuracy = model.accuracy(X_val, y_val)
    print(f"Training loss: {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.2%}; validation accuracy: {val_accuracy:.2%}")
    return {"model": model, "loss_history": history, "initial_loss": initial_loss,
            "final_loss": final_loss, "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cifar", action="store_true",
                        help="Run the neuron, gradient checks, and XOR without loading CIFAR-10")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64],
                        help="Hidden layer widths, for example: --hidden-dims 64 32")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--reg", type=float, default=1e-4)
    options = vars(parser.parse_args())
    skip_cifar = options.pop("skip_cifar")
    run_neuron_example()
    for name, error in run_gradient_checks().items():
        print(f"Gradient check [{name}]: max absolute error={error:.2e}")
    run_xor_demo()
    if not skip_cifar:
        run_cifar10_demo(**options)
