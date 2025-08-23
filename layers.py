import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0/in_dim)
        self.b = np.zeros((1, out_dim))
        self.dw, self.db = None, None
        self.x = None
    
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dw = self.x.T @ grad_out / self.x.shape[0]
        self.db = np.mean(grad_out, axis=0, keepdims=True)
        return grad_out @ self.W.T
    
class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(float)
        return x * self.mask
    
    def backward(self, grad_out):
        return grad_out * self.mask
    
class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_true = y_true
        N = logits.shape[0]
        loss = -np.log(self.probs[np.arange(N), y_true] + 1e-7).mean()
        return loss
    
    def backward(self):
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y_true] -= 1 
        return grad / N 
