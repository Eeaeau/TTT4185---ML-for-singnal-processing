# from tensorflow import keras
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
from tensorflow import keras
import sys


print(sys.executable)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# # Define function
# def f(x):
#     return x ** 2 + np.cos(20 * x) * np.sign(x)


# # Setup some simulation parameters
# # Number of observations
# N = 5000

# # Plot a "clean" version of the relationship between x and y
# plt.figure(figsize=(10, 8))
# x = np.linspace(-2, 2, N)
# plt.plot(x, f(x))
# plt.show()


# # Ceate a model with a single hidden layer. Note that input and output has
# # dimension one
# M = 512
# model = keras.Sequential(
#     [keras.layers.Dense(M, activation=tf.nn.relu, input_dim=1), keras.layers.Dense(1)]
# )

# model.summary()
# # Train the model
# model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
