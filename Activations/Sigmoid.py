import numpy as np
from Base import *

class Sigmoid(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.parameters = []
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs =  1/(1 + np.exp(-np.clip(inputs,-500,500)))
        return self.outputs
    
    def backward(self, output_gradient):
        return output_gradient * (self.outputs * (1- self.outputs))