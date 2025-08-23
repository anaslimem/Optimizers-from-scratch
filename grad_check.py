import numpy as np

def grad_check(model, X, y, eps=1e-5):
    loss, _ = model.forward(X, y)
    model.backward()
    for p, g in model.params_and_grads():
        if g is None:
            print(f"Warning: Gradient is None for parameter shape {p.shape}")
            continue
        idx = np.unravel_index(np.random.randint(p.size), p.shape)
        old = p[idx]
        p[idx] = old + eps
        loss_plus, _ = model.forward(X, y)
        p[idx] = old - eps
        loss_minus, _ = model.forward(X, y)
        p[idx] = old
        num_grad = (loss_plus - loss_minus) / (2*eps)
        print("grad_check:", num_grad, g[idx])
