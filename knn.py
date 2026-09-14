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
Xte = Xte.reshape(Xte.shape[0], -1)

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
        self.Xtr = X 
        self.ytr = y 

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum(X[i] - self.Xtr[j])**2) 
        return dists

    def pred(self, X, k=1):
        dists = self.compute_distances(X)
        return self.pred_labels(dists, k)

    def pred_labels(self, dist, k=1):
        num_test = dist.shape[0]
        predictions = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            neearest_indices = np.argsort(dist[i])[:k]
            nearest_labels = self.ytr[neearest_indices]
            counts = np.bincount(nearest_labels)
            predictions[i] = np.argmax(counts)
        return predictions

knn = KNN()
knn.fit(Xtr, ytr)
pred = knn.pred(Xte, k=5)
accuracy = np.mean(pred == yte)
print(f"accuracy: {accuracy:.4f}")





