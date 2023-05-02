import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# dot product between inputs and weights
output = np.dot(weights, inputs) + bias

print(output)
