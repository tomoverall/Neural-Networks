import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        # we multiply the Gaussian distribution by 0.01 to make the values smaller and easier to fit data to in training (variance of 1 centred around 0)
        # the values will start off small enough that they wont affect training.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # np random takes dimension sizes as parameters and creates the output array with this shape.
        self.biases = np.zeros((1, n_neurons))
        # np zero takes a desired array shape as an argument and returns an array of that shape filled with zeros. Row vector so transposition is not needed.

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
