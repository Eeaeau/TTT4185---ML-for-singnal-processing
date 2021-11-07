import numpy as np
import matplotlib.pyplot as plt

# %matplotlib notebook

import tensorflow as tf
from tensorflow import keras

import os
import sys

# os.path.dirname(sys.executable)
# os.path.dirname(sys.executable)

# sys.executable = "C:/ProgramData/Miniconda3/envs/tf/python.exe"

os.environ['KMP_DUPLICATE_LIB_OK']='True'
print(sys.executable)

# Ceate a model with a single hidden layer. Note that input and output has
# dimension one
M = 512
model = keras.Sequential(
    [keras.layers.Dense(M, activation=tf.nn.relu, input_dim=1), keras.layers.Dense(1)]
)

model.summary()
# Train the model
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])


# Define function
def f(x):
    return x ** 2 + np.cos(20 * x) * np.sign(x)


# Setup some simulation parameters
# Number of observations
N = 5000

# Plot a "clean" version of the relationship between x and y
plt.figure(figsize=(10, 8))
x = np.linspace(-2, 2, N)
plt.plot(x, f(x))
plt.show()

# # train the model
# history = model.fit(x, f(x), epochs=1, batch_size=128, verbose=True)

# history
