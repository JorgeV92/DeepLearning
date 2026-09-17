import argparse
import math
from numbers import Real

import numpy as np

from optimization import eval_numerical_gradient, stochastic_gradient_descent


class Value:
    def __init__(self, data, label="", _parents=()):
        if not isinstance(data, Real):
            raise TypeError("Value requires a real scalar")
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        # Keep duplicate edges: x*x contributes two gradients to the same x.
        self._parents = tuple(_parents)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, label={self.label!r})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, _parents=((self, 1.0), (other, 1.0)))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data,
                     _parents=((self, other.data), (other, self.data)))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, exponent):
        if not isinstance(exponent, Real):
            raise TypeError("the exponent must be a constant real scalar")
        exponent = float(exponent)
        if self.data < 0 and not exponent.is_integer():
            raise ValueError("fractional powers require a nonnegative base")
        if self.data == 0 and 0 < exponent < 1:
            raise ValueError("the derivative of this power is undefined at zero")
        result = self.data ** exponent
        derivative = 0.0 if exponent == 0 else exponent * self.data ** (exponent - 1)
        return Value(result, _parents=((self, derivative),))

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def exp(self):
        result = math.exp(self.data)
        return Value(result, _parents=((self, result),))

    def log(self):
        return Value(math.log(self.data), _parents=((self, 1 / self.data),))

    def sigmoid(self):
        exp_negative_abs = math.exp(-abs(self.data))
        result = (1 / (1 + exp_negative_abs) if self.data >= 0
                  else exp_negative_abs / (1 + exp_negative_abs))
        return Value(result, _parents=((self, result * (1 - result)),))

    def relu(self):
        return Value(max(0.0, self.data), _parents=((self, float(self.data > 0)),))

    def backward(self, seed=1.0):
        """Compute gradients of this output, scaled by seed, in reverse order.
        Each node must receive all downstream contributions before passing its
        gradient to its parents. An iterative depth-first traversal orders the
        graph without hitting Python's recursion limit on long expressions.
        """
        if not isinstance(seed, Real):
            raise TypeError("seed must be a real scalar")
        ordered = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, parents_visited = stack.pop()
            if parents_visited:
                ordered.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for parent, _ in node._parents:
                    stack.append((parent, False))

        for node in ordered:
            node.grad = 0.0
        self.grad = float(seed)  # The derivative of the output with respect to itself.
        for node in reversed(ordered):
            for parent, local_derivative in node._parents:
                parent.grad += local_derivative * node.grad


def scalar_chain_rule(x, y, z):
    """Return f=(x+y)*z and its gradients, starting backward with df/df=1."""
    # Forward: save the intermediate result q.
    q = x + y
    value = q * z
    # Backward: multiply gate, then addition gate.
    dvalue = 1.0
    dq = dvalue * z
    dz = dvalue * q
    dx = dq * 1.0
    dy = dq * 1.0
    return value, {"x": dx, "y": dy, "z": dz}


def branched_expression(x, y):
    """Return f=x*y+(x+y)**2 and gradients; both inputs feed two branches."""
    product = x * y
    total = x + y
    squared = total ** 2
    value = product + squared

    dproduct = 1.0
    dsquared = 1.0
    dtotal = dsquared * 2 * total
    dx = dproduct * y
    dy = dproduct * x
    # Add contributions from the second branch instead of overwriting them.
    dx += dtotal
    dy += dtotal
    return value, {"x": dx, "y": dy}


def affine_forward(X, W, b):
    """Compute X @ W + b; cache references to X and W until backward finishes."""
    return X @ W + b, (X, W)


def affine_backward(dout, cache):
    """Return dX (N, D), dW (D, C), db (C,) for upstream dout (N, C)."""
    X, W = cache
    dX = dout @ W.T
    dW = X.T @ dout
    # Forward broadcasts b to every example; backward sums those contributions.
    db = dout.sum(axis=0)
    return dX, dW, db


def sigmoid_forward(x):
    """Compute sigmoid elementwise and cache its output for the derivative."""
    x = np.asarray(x)
    # Equivalent to 1 / (1 + exp(-x)), without overflowing for negative x.
    exp_negative_abs = np.exp(-np.abs(x))
    out = np.where(x >= 0, 1 / (1 + exp_negative_abs),
                   exp_negative_abs / (1 + exp_negative_abs))
    return out, out


def sigmoid_backward(dout, cache):
    """Multiply the upstream gradient by sigmoid's local derivative."""
    out = cache
    return dout * out * (1 - out)


def relu_forward(x):
    """ReLU is a max gate: max(0, x). Cache which inputs were positive."""
    return np.maximum(0, x), x > 0


def relu_backward(dout, cache):
    """Route gradients through positive inputs; choose derivative 0 at x=0."""
    return dout * cache


def softmax_cross_entropy(scores, y):
    """Return mean cross-entropy and dL/dscores for integer class labels y.

    Softmax and cross-entropy form one combined gate. Its backward expression
    simplifies to (probabilities - one_hot_labels) / N.
    """
    N = len(scores)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    normalizer = exp_scores.sum(axis=1, keepdims=True)
    # Compute log probabilities directly, avoiding log(0) after underflow.
    log_probs = shifted - np.log(normalizer)
    loss = -log_probs[np.arange(N), y].mean()
    dscores = exp_scores / normalizer
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return float(loss), dscores


def linear_softmax_loss(W, b, X, y, reg):
    """The lc.py softmax objective, computed using explicit backprop stages.

    Return (loss, dW, db) so the existing GD/SGD functions can use this loss.
    Backprop also computes dX, although training updates only W and b.
    """
    scores, cache = affine_forward(X, W, b)
    loss, dscores = softmax_cross_entropy(scores, y)
    loss += reg * np.sum(W * W)

    dX, dW, db = affine_backward(dscores, cache)
    # W affects both the data loss and regularization: add both gradients.
    dW += 2 * reg * W
    return float(loss), dW, db


def run_gradient_checks():
    """Return max absolute gradient errors on tiny float64 problems.

    For a vector-valued gate, sum(output * upstream) gives a scalar objective
    whose gradient checks the backward pass with that upstream gradient.
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4, 3))
    W = rng.normal(scale=0.1, size=(3, 2))
    b = rng.normal(scale=0.1, size=2)
    upstream = rng.normal(size=(4, 2))
    y = np.array([0, 1, 1, 0])

    # A sigmoid neuron: affine -> sigmoid. Backward traverses in reverse.
    scores, affine_cache = affine_forward(X, W, b)
    _, sigmoid_cache = sigmoid_forward(scores)
    dscores = sigmoid_backward(upstream, sigmoid_cache)
    dX, dW, db = affine_backward(dscores, affine_cache)

    def neuron_objective(X, W, b):
        scores, _ = affine_forward(X, W, b)
        activations, _ = sigmoid_forward(scores)
        return np.sum(activations * upstream)

    _, loss_dW, loss_db = linear_softmax_loss(W, b, X, y, reg=0.1)
    checks = (
        ("sigmoid neuron: X", lambda x: neuron_objective(x, W, b), X, dX),
        ("sigmoid neuron: W", lambda w: neuron_objective(X, w, b), W, dW),
        ("sigmoid neuron: b", lambda bias: neuron_objective(X, W, bias), b, db),
        ("softmax loss: W", lambda w: linear_softmax_loss(w, b, X, y, 0.1)[0], W, loss_dW),
        ("softmax loss: b", lambda bias: linear_softmax_loss(W, bias, X, y, 0.1)[0], b, loss_db),
    )
    errors = {}
    for name, objective, parameter, analytical in checks:
        numerical = eval_numerical_gradient(objective, parameter)
        errors[name] = float(np.max(np.abs(analytical - numerical)))
    return errors


def run_cifar10_demo(num_train=1000, num_iters=200, batch_size=128,
                     learning_rate=1e-2, reg=1e-4):
    """Train the modular softmax classifier using the existing mini-batch SGD."""
    from lc import LinearClassifier
    from utils import load_cifar10

    if not isinstance(num_train, (int, np.integer)) or not 1 <= num_train <= 49000:
        raise ValueError("num_train must be an integer between 1 and 49000")
    X_train, y_train, X_val, y_val, _, _ = load_cifar10()
    indices = np.random.default_rng(42).choice(len(X_train), num_train, replace=False)
    X_train, y_train = X_train[indices], y_train[indices]
    model = LinearClassifier(X_train.shape[1], 10, linear_softmax_loss)
    initial_loss = linear_softmax_loss(model.W, model.b, X_train, y_train, reg)[0]
    print(f"CIFAR-10: {num_train} training images, {num_iters} SGD updates")
    print(f"Initial training loss: {initial_loss:.4f}")
    model.W, model.b, history = stochastic_gradient_descent(
        linear_softmax_loss, model.W, model.b, X_train, y_train,
        learning_rate=learning_rate, reg=reg, num_iters=num_iters,
        batch_size=batch_size,
    )
    final_loss = linear_softmax_loss(model.W, model.b, X_train, y_train, reg)[0]
    train_accuracy = float(model.accuracy(X_train, y_train))
    val_accuracy = float(model.accuracy(X_val, y_val))
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.2%}; validation accuracy: {val_accuracy:.2%}")
    return {
        "model": model,
        "loss_history": history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cifar", action="store_true",
                        help="Run only small examples and checks, without loading CIFAR-10")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--reg", type=float, default=1e-4)
    options = vars(parser.parse_args())
    skip_cifar = options.pop("skip_cifar")

    value, gradients = scalar_chain_rule(-2.0, 5.0, -4.0)
    print(f"f=(x+y)*z: value={value}, gradients={gradients}")
    x, y, z = Value(-2.0, "x"), Value(5.0, "y"), Value(-4.0, "z")
    output = (x + y) * z
    output.backward()
    print(f"Value automatic gradients: x={x.grad}, y={y.grad}, z={z.grad}")
    value, gradients = branched_expression(2.0, -3.0)
    print(f"f=x*y+(x+y)^2: value={value}, gradients={gradients}")
    for name, error in run_gradient_checks().items():
        print(f"Gradient check [{name}]: max absolute error={error:.2e}")
    if not skip_cifar:
        run_cifar10_demo(**options)
