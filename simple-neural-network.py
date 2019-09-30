import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

assert hasattr(tf, "function")

fashion_mnist = tf.keras.datasets.fashion_mnist
(images, targets), (_, _) = fashion_mnist.load_data()

images = images[:10000]
targets = targets[:10000]

print(images.shape)
print(targets.shape)

targets_names = ["t-shirt", "trouser", "Pullover", "Dress", "Coat", "Sandal", "shirt", 
"Sneakers", "Bag", "Ankle boot"]

# plt.imshow(images[11], cmap="binary")
# plt.title(targets_names[targets[11]])
# print(targets[11])
# plt.show()


# Create the model
model = tf.keras.models.Sequential()

# Flatten the image
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))

print("Shape of the image", images[0:1].shape)
model_ouput = model.predict(images[0:1])
print("Shape of the Flattened image", model_ouput.shape)

# Add layers
model.add(tf.keras.layers.Dense(256, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model_ouput = model.predict(images[0:1])
print(model_ouput, targets[0:1])

# model.summary()


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

