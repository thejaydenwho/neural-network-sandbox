import numpy as np
from Base import *

class Dense(Layer):
    def __init__(self, n_inputs, n_outputs, weights = None, biases = None):
        self.inputs = None
        self.outputs = None
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.parameters = []
        if weights is None:
            self.weights = Parameter(0.1 * np.random.randn(n_inputs, n_outputs), None)
        else:
            self.weights = Parameter(weights, None)
        if biases is None:
            self.biases = Parameter(0.1 * np.random.randn(n_outputs), None)
        else:
            self.biases = Parameter(biases, None)
        self.parameters.append(self.weights)
        self.parameters.append(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights.value) + self.biases.value
        return self.outputs
    
    def backward(self, output_gradient):
        dX = output_gradient @ self.weights.value.T
        dW = self.inputs.T @ output_gradient
        db = np.sum(output_gradient, axis=0)
        self.weights.gradient = dW
        self.biases.gradient = db
        return dX
