# list of inputs
inputs = [1.0, 2.0, 3.0, 2.5]
# list of lists of weights
weights = [
    [0.2, 0.8, -0.5, 1.0],  # neuron 1
    [0.5, -0.91, 0.26, -0.5],  # neuron 2
    [-0.26, -0.27, 0.17, 0.87],  # neuron 3
]
# list of biases
biases = [2.0, 3.0, 0.5]

# output of current layer
layer_outputs = []
# for each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # zeroed output of given neuron
    neuron_output = 0
    # for each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # multiply this input by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input * weight
    # add bias
    neuron_output += neuron_bias
    # put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)

print(layer_outputs)
