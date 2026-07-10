import numpy as np
from Base import *

class Flatten(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.parameters = []

    def forward(self, inputs):
        self.inputs = inputs
        (batch_size, _, _, _) = inputs.shape
        outputs = np.reshape(inputs, (batch_size, -1))
        self.outputs = outputs
        return outputs
    
    def backward(self, output_gradient):
        original_shape = self.inputs.shape
        input_gradient = np.reshape(output_gradient, original_shape)
        return input_gradient