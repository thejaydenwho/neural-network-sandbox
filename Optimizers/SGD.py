import numpy as np
from Base import *

class SGD:
    def __init__(self, learn_rate=0.01):
        self.learn_rate = np.float32(learn_rate)

    def update_model(self, sequential):
        for parameter in sequential.parameters():
            grad = parameter.gradient.astype(np.float32)
            parameter.value -= self.learn_rate * grad
    
            