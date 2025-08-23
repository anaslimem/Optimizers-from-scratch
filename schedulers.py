import numpy as np

class ConstantLR:
    def __init__(self, lr): self.lr = lr 
    def get(self, epoch): return self.lr 

class StepDecay:
    def __init__(self, lr, drop=0.5, step=10):
        self.lr, self.drop, self.step = lr, drop, step
    def get(self, epoch): return self.lr * (self.drop ** (epoch // self.step))

class CosineDecay:
    def __init__(self, lr, epochs):
        self.lr, self.epochs = lr, epochs 
    def get(self, epoch):
        return self.lr * 0.5 * (1 + np.cos(np.pi * epoch / self.epochs))
    
class WarmupLR:
    def __init__(self, lr, warmup=5):
        self.lr, self.warmup = lr, warmup
    def get(self, epoch):
        return self.lr * min(1.0, epoch / self.warmup)

        
    