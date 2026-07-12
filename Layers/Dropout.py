import numpy as np
from Base import *

class Dropout(Layer):
    def __init__(self, percent = 0.5, seed = None):
        self.inputs = None
        self.outputs = None
        self.percent = percent
        self.rng = np.random.default_rng(seed)
        self.parameters = []
    
    def forward(self, inputs, training=True):
        self.inputs = np.asarray(inputs, dtype=np.float32)
        if training:
            self.mask = (self.rng.random(self.inputs.shape) >= self.percent).astype(np.float32)
            self.mask /= (1-self.percent)
            self.outputs = self.inputs * self.mask
        else: 
            self.outputs = self.inputs
        return self.outputs
    
    def backward(self, output_gradient):
        output_gradient = np.asarray(output_gradient, dtype=np.float32)
        return output_gradient * self.mask

