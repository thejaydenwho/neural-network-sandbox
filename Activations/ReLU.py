import numpy as np
from Base import *

class ReLU(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.parameters = []

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(inputs,0)
        return self.outputs
    
    def backward(self, output_gradient):
        return output_gradient * (self.inputs > 0)
    
