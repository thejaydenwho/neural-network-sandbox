import numpy as np
from Base import *

class Sigmoid(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.parameters = []
    
    def forward(self, inputs, training=None):
        self.inputs = np.asarray(inputs, dtype=np.float32)
        clipped_inputs = np.clip(self.inputs, np.float32(-500.0), np.float32(500.0))
        self.outputs = np.float32(1.0) / (np.float32(1.0) + np.exp(-clipped_inputs))
        return self.outputs 
    
    def backward(self, output_gradient):
        output_gradient = np.asarray(output_gradient, dtype=np.float32)
        return output_gradient * (self.outputs * (np.float32(1.0) - self.outputs))