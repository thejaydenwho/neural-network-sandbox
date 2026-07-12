import numpy as np 
from Base import * 

class Softmax(Layer): 
    def __init__(self): 
        self.inputs = None 
        self.outputs = None 
        self.parameters = [] 

    def forward(self, inputs, training=None): 
        self.inputs = np.asarray(inputs, dtype=np.float32)
        shift_inputs = self.inputs - np.max(self.inputs, axis=1, keepdims=True, initial=None)
        exp_values = np.exp(shift_inputs)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True, dtype=np.float32)
        return self.outputs 

    def backward(self, output_gradient):
        return output_gradient