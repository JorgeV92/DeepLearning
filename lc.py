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

def softmax_loss(W, b, X, y, reg):
    N = X.shape[0]
    scores = X @ W + b
    scores -= np.max(scores, axis=1, keepdims=True) 
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_probs = probs[np.arange(N), y]
    loss = -np.mean(np.log(corect_probs + 1e-12))
    loss += reg * np.sum(W*W)
    # Gradient 
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N 
    dW = X.T @ dscores
    db = dscores.sum(axis=0)
    dW += 2 * reg * W 
    return loss, dW, db
