import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        # we multiply the Gaussian distribution by 0.01 to make the values smaller and easier to fit data to in training (variance of 1 centred around 0)
        # the values will start off small enough that they wont affect training.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # np random takes dimension sizes as parameters and creates the output array with this shape.
        self.biases = np.zeros((1, n_neurons))
        # np zero takes a desired array shape as an argument and returns an array of that shape filled with zeros. Row vector so transposition is not needed.

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])
