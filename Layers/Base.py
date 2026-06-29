import numpy as np

class Layer:
    def __init__(self):
        # Each layer must remember its inputs and outputs
        # during the forward pass to use them in the backward pass
        self.inputs = None
        self.outputs = None
    def forward(self, inputs):
        # Takes input data, processes it, and returns the output
        # Must be overridden by the actual layer (Dense, ReLU, etc.)
        raise NotImplementedError("Forgot to implement forward pass")
    
    def backward(self, output_gradient):
        # Takes the gradient from layer ahead
        # Calculates partial derivatives and returns gradient for layer behind
        raise NotImplementedError("Forgot to implement backward pass")


class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        # Adds a new layer to the end of our neural network
        self.layers.append(layer)
    
    def forward(self, inputs):
        # Passes the data forward through every layer in order
        current_signal = inputs
        for layer in self.layers:
            current_signal = layer.forward(current_signal)
        return current_signal
    def backward(self, output_gradient):
        # Passes the output gradient backward through every layer in REVERSE order
        current_gradient = output_gradient
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)
        return current_gradient