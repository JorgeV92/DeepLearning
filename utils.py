import numpy as np 
from torchvision.datasets import CIFAR10

def load_cifar10():
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

    X_train = X[:49000]
    y_train = y[:49000]

    X_val = X[49000:]
    y_val = y[49000:]

    mean_image = X_train.mean(axis=0, keepdims=True)

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test
 