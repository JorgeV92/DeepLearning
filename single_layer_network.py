import numpy as np 
from backpropagation import softmax_cross_entropy 

class SingleLayerNetwork:
    def __init__(self):
        self.W = np.array([[0.1, -0.1], 
                           [0.2, 0.2]])
        self.b = np.zeros(2)
    def forward(self, X):
        return X @ self.W + self.b 
    def train_step(self, X, y, lr=0.1):
        scores = self.forward(X)
        loss, dscores = softmax_cross_entropy(scores, y)
        dW = X.T @ dscores 
        db = dscores.sum(axis=0)
        self.W -= lr * dW 
        self.b -= lr * db 
        return loss 

def run_example():
    X = np.array([[2.0, 1.0]]) 
    y = np.array([1])           
    network = SingleLayerNetwork()

    print("Architecture: 2 input features -> 2 output neurons (one layer)")
    print("Starting weights:\n", network.W)
    print("Starting biases:", network.b)
    scores = network.forward(X)
    print("\nInput:", X)
    print("Scores before training:", scores)
    print("Predicted class:", scores.argmax(axis=1), "correct class:", y)

    loss_before = network.train_step(X, y)
    new_scores = network.forward(X)
    loss_after, _ = softmax_cross_entropy(new_scores, y)
    print("\nAfter one gradient-descent step:")
    print("Updated weights:\n", network.W)
    print("Updated biases:", network.b)
    print("Scores:", new_scores)
    print("Predicted class:", new_scores.argmax(axis=1))

if __name__ == '__main__':
    run_example()