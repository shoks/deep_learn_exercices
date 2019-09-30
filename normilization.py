import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler

assert hasattr(tf, "function")

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()

print(images.mean())
print(images.std())

# Flatten
images = images.reshape(-1, 784)
images = images.astype(float)

scaler = StandardScaler()
images = scaler.fit_transform(images)

print(images.mean())
print(images.std())


# Create the model
model = tf.keras.models.Sequential()

# Add layers
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model_ouput = model.predict(images[0:1])

model.summary()

print(model_ouput, targets[0:1])

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)


history = model.fit(images, targets, epochs=10)

loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

plt.plot(loss_curve)
plt.title("loss")
plt.show()

plt.plot(acc_curve)
plt.title("accuracy")
plt.show()
