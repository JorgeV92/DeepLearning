import numpy as np 
from torchvision.datasets import CIFAR10

def load_cifar10(num_val=1000):
    """Load normalized CIFAR-10 data; use num_val=0 for all training images."""
    train_data = CIFAR10(root="./data",train=True,download=True)
    test_data = CIFAR10(root="./data",train=False,download=True)

    X = train_data.data.astype(np.float32)
    y = np.array(train_data.targets)

    X_test = test_data.data.astype(np.float32)
    y_test = np.array(test_data.targets)

    X = X.reshape(X.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X /= 255.0
    X_test /= 255.0

    if not 0 <= num_val < X.shape[0]:
        raise ValueError("num_val must leave at least one training image")

    num_train = X.shape[0] - num_val
    X_train = X[:num_train]
    y_train = y[:num_train]

    X_val = X[num_train:]
    y_val = y[num_train:]

    mean_image = X_train.mean(axis=0, keepdims=True)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test
