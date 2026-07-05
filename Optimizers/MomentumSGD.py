import numpy as np
from Base import *

class MomentumSGD:
    def __init__(self, learn_rate, momentum):
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.velocity_dict = {}

    def update_model(self, sequential):
        for parameter in sequential.parameters():
            key = id(parameter)
            if key not in self.velocity_dict:
                self.velocity_dict[key] = np.zeros_like(parameter.value)
            self.velocity_dict[key] = (self.momentum * self.velocity_dict[key]) - (self.learn_rate * parameter.gradient)
            parameter.value += self.velocity_dict[key]


    

