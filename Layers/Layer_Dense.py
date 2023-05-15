import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

nnfs.init()


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

    def backward(self, dvalues):
        self.dweights = np.dot(self.dinputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        # calculate sample losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # number of samples in a batch
        samples = len(y_pred)
        # clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()

loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)

predictions = np.argmax(activation2.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print("Accuracy: ", accuracy)
