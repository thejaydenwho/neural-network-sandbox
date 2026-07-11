import numpy as np
from Base import *

class ReLU(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.parameters = []

    def forward(self, inputs):
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.outputs = np.maximum(inputs,0.0)
        return self.outputs
    
    def backward(self, output_gradient):
        output_gradient = np.asarray(output_gradient, dtype=np.float32)
        relu_mask = (self.inputs > 0).astype(np.float32)
        return output_gradient * relu_mask
    
