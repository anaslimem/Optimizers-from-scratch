import numpy as np

def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def train(model, optimizer, scheduler, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
    history = {"loss": [], "acc": []}
    n_batches = X_train.shape[0] // batch_size
    for epoch in range(1, epochs+1):
        lr = scheduler.get(epoch)
        optimizer.lr = lr
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]
        for i in range(n_batches):
            xb = X_train[i*batch_size:(i+1)*batch_size]
            yb = y_train[i*batch_size:(i+1)*batch_size]
            loss, logits = model.forward(xb, yb)
            model.backward()
            optimizer.step(model.params_and_grads())
        test_loss, test_logits = model.forward(X_test, y_test)
        acc = accuracy(test_logits, y_test)
        history["loss"].append(test_loss)
        history["acc"].append(acc)
        print(f"Epoch {epoch}: loss={test_loss:.4f}, acc={acc:.4f}, lr={lr:.5f}")
    return history