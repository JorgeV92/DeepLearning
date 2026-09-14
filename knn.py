import numpy as np 
from torchvision.datasets import CIFAR10 

train_dataset = CIFAR10(root="./data", train=True, download=True)
test_dataset = CIFAR10(root="./data", train=False, download=True)

Xtr = train_dataset.data 
ytr = np.array(train_dataset.targets)

Xte = test_dataset.data 
yte = np.array(test_dataset.targets)

print(Xtr.shape); print(ytr.shape) 
print(Xte.shape); print(yte.shape)

Xtr = Xtr.reshape(Xtr.shape[0], -1)
Xte = Xte.reshape(Xtr.shape[0], -1)

Xtr = Xtr.astype(np.float32)
Xte = Xte.astype(np.float32)

print(Xtr.shape)

Xtr = Xtr[:5000]
ytr = ytr[:5000]

Xte = Xte[:500]
yte = yte[:500]

class KNN:
    def __init__(self) -> None:
        self.Xtr = None 
        self.ytr = None 

    def fit(self, X, y):
        pass 

    def compute_distances(self, X):
        pass 

    def pred(self, X, k=1):
        pass 

    def pred_labels(self, dist, k=1):
        pass 





