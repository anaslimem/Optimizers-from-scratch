import sys
import tensorflow as tf
from model import MLP
from optimizers import SGD, Momentum, RMSProp, Adam, AdamW
from schedulers import ConstantLR, StepDecay, CosineDecay, WarmupLR
from train import train
import matplotlib.pyplot as plt
import numpy as np

# Redirect stdout to a file to capture all training output
f = open("training_log.txt", "w")
sys.stdout = f

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

epochs = 10

optimizers = {
    "SGD": SGD(lr=0.01),
    "Momentum": Momentum(lr=0.01),
    "RMSProp": RMSProp(lr=0.0001),
    "Adam": Adam(lr=0.001),
    "AdamW": AdamW(lr=0.001, wd=0.01)
}

schedulers = {
    "ConstantLR": ConstantLR(lr=0.001),
    "StepDecay": StepDecay(lr=0.001, drop=0.5, step=3),
    "CosineDecay": CosineDecay(lr=0.001, epochs=epochs),
    "WarmupLR": WarmupLR(lr=0.001, warmup=3)
}

results = {}

for opt_name, optimizer in optimizers.items():
    for sched_name, scheduler in schedulers.items():
        print(f"\n--- Training with Optimizer: {opt_name}, Scheduler: {sched_name} ---")
        model = MLP()
        history = train(model, optimizer, scheduler, X_train, y_train, X_test, y_test, epochs=epochs)
        results[(opt_name, sched_name)] = history

        # Plotting the learning rate schedule
        lrs = [scheduler.get(epoch) for epoch in range(1, epochs + 1)]
        plt.figure()
        plt.plot(range(1, epochs + 1), lrs)
        plt.title(f"LR Schedule for {opt_name} with {sched_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.savefig(f"lr_schedule_{opt_name}_{sched_name}.png")
        plt.close()

        # Plotting loss and accuracy
        plt.figure()
        plt.plot(history["loss"], label="loss")
        plt.plot(history["acc"], label="accuracy")
        plt.title(f"Training History for {opt_name} with {sched_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(f"training_history_{opt_name}_{sched_name}.png")
        plt.close()

sys.stdout.close()
sys.stdout = sys.__stdout__

print("Training and plotting complete. Check the generated PNG files and training_log.txt.")

