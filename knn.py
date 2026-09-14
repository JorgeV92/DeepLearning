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
