import numpy as np
from utils import load_cifar10

def svm_loss(W, b, X, y, reg):
    N = X.shape[0]
    scores = X @ W + b 
    correct_scores = scores[np.arange(N), y][:, None]
    margins = np.maximum(0, scores - correct_scores + 1.0)
    margins[np.arange(N), y] = 0
    loss = margins.sum() / N 
    loss += reg * np.sum(W * W)
    # Gradient 
    dscores = (margins>0).astype(np.float32)
    viol = dscores.sum(axis=1)
    dscores[np.arange(N), y] = -viol 
    dscores /= N 
    dW = X.T @ dscores
    db = dscores.sum(axis=0)
    dW += 2 * reg * W 
    return loss, dW, db 

def softmax_loss(W, b, X, y, reg):
    N = X.shape[0]
    scores = X @ W + b
    scores -= np.max(scores, axis=1, keepdims=True) 
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    loss = np.mean(np.log(np.sum(exp_scores, axis=1)) - scores[np.arange(N), y])
    loss += reg * np.sum(W*W)
    # Gradient 
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N 
    dW = X.T @ dscores
    db = dscores.sum(axis=0)
    dW += 2 * reg * W 
    return loss, dW, db


class LinearClassifier:
    def __init__(self, input_dim, num_classes, loss_fn):
        self.loss_fn = loss_fn 
        rng = np.random.default_rng(42)
        self.W = (0.001 * rng.standard_normal((input_dim, num_classes))).astype(np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

    def train(self, X, y, learning_rate=1e-2, reg=1e-4, num_iters=1000,
              batch_size=256, num_epochs=None):
        """Run SGD; num_epochs overrides num_iters to visit every image each epoch."""
        rng = np.random.default_rng(42)
        N = X.shape[0]
        if N == 0:
            raise ValueError("training data must contain at least one image")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_epochs is not None:
            if num_epochs <= 0:
                raise ValueError("num_epochs must be positive")
            batches_per_epoch = (N + batch_size - 1) // batch_size
            num_iters = num_epochs * batches_per_epoch

        for it in range(num_iters):
            if num_epochs is None:
                indices = rng.choice(N, size=min(batch_size, N), replace=False)
            else:
                batch_index = it % batches_per_epoch
                if batch_index == 0:
                    epoch_indices = rng.permutation(N)
                start = batch_index * batch_size
                indices = epoch_indices[start:start + batch_size]
            X_batch = X[indices]
            y_batch = y[indices]
            loss, dW, db = self.loss_fn( self.W, self.b, X_batch, y_batch, reg)
            # SGD update
            self.W -= learning_rate * dW
            self.b -= learning_rate * db
            if it % 100 == 0:
                print(
                    f"iteration {it:4d} "
                    f"loss = {loss:.4f}")
    def pred(self, X):
        scores = X @ self.W + self.b 
        return np.argmax(scores, axis=1)

    def accuracy(self, X, y):
        predictions = self.pred(X)
        return np.mean(predictions == y)


def train_svm_cifar10(num_train=None, num_iters=1000, batch_size=256):
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

    if num_train is not None:
        X_train = X_train[:num_train]
        y_train = y_train[:num_train]

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    svm = LinearClassifier(input_dim=3072,num_classes=10,loss_fn=svm_loss)
    svm.train(X_train, y_train, learning_rate=1e-2, reg=1e-4, num_iters=num_iters, batch_size=batch_size)

    print("SVM train accuracy:", svm.accuracy(X_train, y_train))
    print("SVM validation accuracy:", svm.accuracy(X_val, y_val))
    print("SVM test accuracy:", svm.accuracy(X_test, y_test))
    return svm


def train_softmax_cifar10(num_train=None, num_iters=1000, batch_size=256):
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()

    if num_train is not None:
        X_train = X_train[:num_train]
        y_train = y_train[:num_train]

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    softmax = LinearClassifier(input_dim=3072, num_classes=10, loss_fn=softmax_loss)
    softmax.train(X_train, y_train, learning_rate=1e-2, reg=1e-4, num_iters=num_iters, batch_size=batch_size)

    print("Softmax train accuracy:", softmax.accuracy(X_train, y_train))
    print("Softmax validation accuracy:", softmax.accuracy(X_val, y_val))
    print("Softmax test accuracy:", softmax.accuracy(X_test, y_test))
    return softmax


def train_full_cifar10(num_epochs=10, batch_size=256, learning_rate=1e-2, reg=1e-4):
    """Train SVM and softmax on all 50,000 training images and predict test labels.
    Returns a dict keyed by "svm" and "softmax", each containing the trained
    model, test predictions, train accuracy, and test accuracy. The 10,000
    test images are used only for evaluation.
    """
    X_train, y_train, _, _, X_test, y_test = load_cifar10(num_val=0)
    print(f"Training images: {len(y_train)}; test images: {len(y_test)}")
    results = {}
    for name, loss_fn in (("svm", svm_loss), ("softmax", softmax_loss)):
        print(f"\nTraining {name.upper()} for {num_epochs} epochs")
        model = LinearClassifier(
            input_dim=X_train.shape[1], num_classes=10, loss_fn=loss_fn
        )
        model.train(
            X_train, y_train, learning_rate=learning_rate, reg=reg,
            batch_size=batch_size, num_epochs=num_epochs,
        )

        predictions = model.pred(X_test)
        train_accuracy = float(model.accuracy(X_train, y_train))
        test_accuracy = float(np.mean(predictions == y_test))
        results[name] = {
            "model": model,
            "predictions": predictions,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
        print(f"{name.upper()} train accuracy: {train_accuracy:.4f}")
        print(f"{name.upper()} test accuracy: {test_accuracy:.4f}")
        print(f"{name.upper()} test predictions (first 10): {predictions[:10]}")

    print(f"\nActual test labels (first 10): {y_test[:10]}")
    return results


if __name__ == '__main__':
    # svm = train_svm_cifar10()
    # softmax = train_softmax_cifar10()
    results = train_full_cifar10()
