import numpy as np
from Base import *

class MomentumSGD:
    def __init__(self, learn_rate=0.01, momentum=0.9):
        self.learn_rate = np.float32(learn_rate)
        self.momentum = np.float32(momentum)
        self.velocity_dict = {}

    def update_model(self, sequential):
        for parameter in sequential.parameters():
            grad = parameter.gradient.astype(np.float32)
            key = id(parameter)
            if key not in self.velocity_dict:
                self.velocity_dict[key] = np.zeros(parameter.gradient.shape, dtype=np.float32)
            self.velocity_dict[key] = (self.momentum * self.velocity_dict[key]) - (self.learn_rate * grad)
            parameter.value += self.velocity_dict[key]


    

