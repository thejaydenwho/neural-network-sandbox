import numpy as np
from Base import *

class Dense(Layer):
    def __init__(self, n_inputs, n_outputs, weights = None, biases = None, seed = None):
        self.inputs = None
        self.outputs = None
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.rng = np.random.default_rng(seed)
        self.parameters = []
        if weights is None:
            self.weights = Parameter(self.rng.normal(loc=0.0, scale=0.1, size=(n_inputs, n_outputs)).astype(np.float32), None)
        else:
            self.weights = Parameter(np.array(weights, dtype=np.float32), None)
        if biases is None:
            self.biases = Parameter(self.rng.normal(loc=0.0, scale=0.1, size=(n_outputs)).astype(np.float32), None)
        else:
            self.biases = Parameter(np.array(biases, dtype=np.float32), None)
        self.parameters.append(self.weights)
        self.parameters.append(self.biases)

    def forward(self, inputs, training=None):
        self.inputs = np.array(inputs, dtype=np.float32)
        self.outputs = np.dot(self.inputs, self.weights.value) + self.biases.value
        return self.outputs
    
    def backward(self, output_gradient):
        output_gradient = np.asarray(output_gradient, dtype=np.float32)
        dX = output_gradient @ self.weights.value.T
        dW = self.inputs.T @ output_gradient
        db = np.sum(output_gradient, axis=0, dtype=np.float32)
        self.weights.gradient = dW
        self.biases.gradient = db
        return dX
