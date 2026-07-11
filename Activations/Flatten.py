import numpy as np
from Base import *

class Flatten(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.parameters = []

    def forward(self, inputs):
        self.inputs = np.asarray(inputs, dtype=np.float32)
        batch_size = self.inputs.shape[0]
        self.outputs = np.reshape(inputs, (batch_size, -1))
        return self.outputs 
    
    def backward(self, output_gradient):
        output_gradient = np.asarray(output_gradient, dtype=np.float32)
        original_shape = self.inputs.shape
        input_gradient = np.reshape(output_gradient, original_shape)
        return input_gradient