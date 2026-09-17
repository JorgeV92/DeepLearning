""" 
X has shape (N, D), W has shape (D, C), and b has shape (C,).
A loss function takes (W, b, X, y, reg) and returns (loss, dW, db).
The update rule is always: parameter -= learning_rate * gradient.
"""
import argparse

import numpy as np

def eval_numerical_gradient(f, x, h=1e-5):
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    x = np.array(x, dtype=np.float64, copy=True)
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + h
        loss_plus = f(x)
        x[index] = original - h
        loss_minus = f(x)
        x[index] = original
        grad[index] = (loss_plus - loss_minus) / (2 * h)
    return grad


def _validate_training_data(X, y, learning_rate, reg, num_iters):
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X must be a nonempty (N, D) array")
    if y.shape != (len(X),):
        raise ValueError("y must have shape (N,)")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")
    if not isinstance(num_iters, (int, np.integer)) or num_iters <= 0:
        raise ValueError("num_iters must be a positive integer")


def gradient_descent(loss_fn, W, b, X, y, learning_rate=1e-2,
                     reg=1e-4, num_iters=100):
    _validate_training_data(X, y, learning_rate, reg, num_iters)
    W, b = W.copy(), b.copy()
    loss_history = []
    for _ in range(num_iters):
        loss, dW, db = loss_fn(W, b, X, y, reg)
        loss_history.append(float(loss))
        # The gradient points uphill, so subtract it to step downhill.
        W -= learning_rate * dW
        b -= learning_rate * db
    return W, b, loss_history


def stochastic_gradient_descent(loss_fn, W, b, X, y, learning_rate=1e-2,
                                reg=1e-4, num_iters=100, batch_size=1, seed=42):
    """Use a random batch for each update; return (W, b, loss_history).
    batch_size=1 is single-example SGD; larger batches give mini-batch SGD.
    """
    _validate_training_data(X, y, learning_rate, reg, num_iters)
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    rng = np.random.default_rng(seed)
    W, b = W.copy(), b.copy()
    loss_history = []
    for _ in range(num_iters):
        indices = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
        X_batch, y_batch = X[indices], y[indices]
        loss, dW, db = loss_fn(W, b, X_batch, y_batch, reg)
        loss_history.append(float(loss))
        # Same update as GD; only the examples used for the gradient differ.
        W -= learning_rate * dW
        b -= learning_rate * db
    return W, b, loss_history


def run_cifar10_demo(loss="svm", num_train=1000, num_iters=100,
                     batch_size=128, learning_rate=1e-2, reg=1e-4):
    from lc import LinearClassifier, softmax_loss, svm_loss
    from utils import load_cifar10

    if loss not in ("svm", "softmax"):
        raise ValueError("loss must be 'svm' or 'softmax'")
    if not isinstance(num_train, (int, np.integer)) or not 1 <= num_train <= 49000:
        raise ValueError("num_train must be an integer between 1 and 49000")
    loss_fn = svm_loss if loss == "svm" else softmax_loss
    X_train, y_train, X_val, y_val, _, _ = load_cifar10()
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_train), size=num_train, replace=False)
    X_train, y_train = X_train[indices], y_train[indices]
    initial = LinearClassifier(X_train.shape[1], 10, loss_fn)

    X_small = X_train[:4, :5].astype(np.float64)
    y_small = y_train[:4]
    W_small = initial.W[:5].astype(np.float64)
    b_small = initial.b.astype(np.float64)
    _, dW, db = loss_fn(W_small, b_small, X_small, y_small, reg)
    numerical_dW = eval_numerical_gradient(
        lambda W: loss_fn(W, b_small, X_small, y_small, reg)[0], W_small
    )
    numerical_db = eval_numerical_gradient(
        lambda b: loss_fn(W_small, b, X_small, y_small, reg)[0], b_small
    )
    print(f"Gradient check (max absolute error): "
          f"W={np.max(np.abs(dW - numerical_dW)):.2e}, "
          f"b={np.max(np.abs(db - numerical_db)):.2e}")

    initial_loss = float(loss_fn(initial.W, initial.b, X_train, y_train, reg)[0])
    print(f"{loss.upper()}: {num_train} training images, {num_iters} updates each")
    print(f"Initial training loss: {initial_loss:.4f}")
    print(f"GD uses {num_train} images/update; "
          f"SGD uses {min(batch_size, num_train)} images/update")
    results = {}
    for name, optimizer, options in (
        ("gd", gradient_descent, {}),
        ("sgd", stochastic_gradient_descent, {"batch_size": batch_size}),
    ):
        W, b, history = optimizer(
            loss_fn, initial.W, initial.b, X_train, y_train,
            learning_rate=learning_rate, reg=reg, num_iters=num_iters, **options,
        )
        model = LinearClassifier(X_train.shape[1], 10, loss_fn)
        model.W, model.b = W, b
        final_loss = float(loss_fn(W, b, X_train, y_train, reg)[0])
        train_accuracy = float(model.accuracy(X_train, y_train))
        val_accuracy = float(model.accuracy(X_val, y_val))
        results[name] = {
            "model": model,
            "loss_history": history,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }
        print(f"{name.upper()}: final training loss={final_loss:.4f}, "
              f"train accuracy={train_accuracy:.2%}, "
              f"validation accuracy={val_accuracy:.2%}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=("svm", "softmax"), default="svm")
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--reg", type=float, default=1e-4)
    run_cifar10_demo(**vars(parser.parse_args()))
