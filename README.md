# Deep Learning 

A list of programs in python for machine learning and deep learning.

Run both linear classifiers on the full CIFAR-10 training set:

```bash
conda activate dl
python lc.py
```

This trains separate SVM and softmax models for 10 epochs on all 50,000 training
images, then prints accuracy and sample predictions on the separate 10,000-image
test set. Every training image is used once per epoch. Data is downloaded to the
ignored `data/` directory if needed.

To customize training and access all test predictions:

```python
from lc import train_full_cifar10

results = train_full_cifar10(num_epochs=10, batch_size=256)
svm_predictions = results["svm"]["predictions"]
softmax_predictions = results["softmax"]["predictions"]
```

Each result also contains `model`, `train_accuracy`, and `test_accuracy`.
The individual `train_svm_cifar10()` and `train_softmax_cifar10()` functions
still use the default 49,000/1,000 training/validation split.

## Optimization: gradient descent and SGD

```python
loss, dW, db = loss_fn(W, b, X_batch, y_batch, reg)
W -= learning_rate * dW
b -= learning_rate * db
```

The gradient describes how loss changes with each parameter. Subtracting it
moves downhill locally; the learning rate determines the step size. A step
that is too large can increase the loss. We update both weights and biases;
the existing loss functions regularize only the weights.

| Function / setting | Examples used for each update |
| --- | --- |
| `gradient_descent(...)` | Entire supplied training set |
| `stochastic_gradient_descent(..., batch_size=1)` | One randomly selected image (SGD) |
| `stochastic_gradient_descent(..., batch_size=128)` | 128 randomly selected images (mini-batch SGD) |

`num_iters` counts parameter updates. SGD samples a fresh batch each time;
it does not guarantee visiting every image, unlike `lc.py`'s epoch mode.
Both functions return `(W, b, loss_history)` and copy the starting parameters,
making it easy to compare runs from the same initialization. Each history
entry is the loss **before** an update: GD records full training loss, while
SGD records batch loss, which can be noisy even when training is working.

Run the comparison on a reproducible subset of 1,000 CIFAR-10 training images:

```bash
conda activate dl
python optimization.py
python optimization.py --loss softmax
python optimization.py --batch-size 1 --num-iters 1000
python optimization.py --learning-rate 0.001
```

Each run checks numerical versus analytical gradients on a tiny problem,
then trains GD and SGD from identical weights. It prints final losses on the
same training subset and accuracy on the 1,000-image validation split. The
number of updates is equal, but GD processes more images per update; this is
not an equal-compute benchmark. Use validation results to experiment with
learning rates and batch sizes. The default settings are learning examples,
not tuned CIFAR-10 hyperparameters.

You can also use the functions directly with the arrays from `load_cifar10()`:

```python
from lc import LinearClassifier, softmax_loss
from optimization import stochastic_gradient_descent
from utils import load_cifar10

X_train, y_train, X_val, y_val, _, _ = load_cifar10()
model = LinearClassifier(X_train.shape[1], 10, softmax_loss)
model.W, model.b, loss_history = stochastic_gradient_descent(
    softmax_loss, model.W, model.b, X_train, y_train,
    learning_rate=0.01, num_iters=1000, batch_size=128,
)
print("Validation accuracy:", model.accuracy(X_val, y_val))
```

`eval_numerical_gradient(f, x)` estimates derivatives by perturbing one
parameter at a time: `(f(x + h) - f(x - h)) / (2 * h)` along that coordinate.
Use it on tiny arrays to check your analytical gradients; it needs two loss
evaluations per parameter, so it is much too expensive for training. SVM
checks can disagree where a perturbation crosses a hinge-loss kink.
