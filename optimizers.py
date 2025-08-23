import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def step(self, params_grads):
        for p, g in params_grads:
            p -= self.lr * g

class Momentum:
    def __init__(self, lr=0.01, mu=0.9):
        self.lr, self.mu = lr, mu
        self.v = {}
    
    def step(self, params_grads):
        for i,(p, g) in enumerate(params_grads):
            if i not in self.v:
                self.v[i] = np.zeros_like(g)
            self.v[i] = self.mu * self.v[i] - self.lr * g
            p += self.v[i]

class RMSProp:
    def __init__(self, lr=0.0001, beta=0.9, eps=1e-8):
        self.lr, self.beta, self.eps = lr, beta, eps
        self.s = {}

    def step(self, params_grads):
        for i, (p,g) in enumerate(params_grads):
            if i not in self.s:
                self.s[i] = np.zeros_like(g)
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (g ** 2)
            p -= self.lr * g / (np.sqrt(self.s[i]) + self.eps)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0
    
    def step(self, params_grads):
        self.t += 1
        for i, (p,g) in enumerate (params_grads):
            if i not in self.m:
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
class AdamW(Adam):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.wd = wd

    def step(self, params_grads):
        self.t += 1
        for i, (p,g) in enumerate (params_grads):
            if i not in self.m:
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p)
    
