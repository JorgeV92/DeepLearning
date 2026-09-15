import numpy as np 

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

    