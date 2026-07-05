import numpy as np
from Base import *

class Softmax(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.parameters = []

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1, keepdims=True)
        self.outputs = probabilities
        return self.outputs
    
    def backward(self, output_gradient):
        return output_gradient
