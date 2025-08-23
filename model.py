from layers import Linear,ReLU,SoftmaxCrossEntropy
class MLP:
    def __init__(self):
        self.layers= [
            Linear(784, 256), ReLU(),
            Linear(256, 10)
        ]
        self.loss_fn = SoftmaxCrossEntropy()
    
    def forward(self, x, y):
        out = x 
        for layer in self.layers:
            out = layer.forward(out)
        loss = self.loss_fn.forward(out, y)
        return loss, out 
    
    def backward(self):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def params_and_grads(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                yield layer.W, layer.dw
                yield layer.b, layer.db
            