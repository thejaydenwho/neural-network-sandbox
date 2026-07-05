import numpy as np
from Base import *

class SGD:
    def __init__(self, learn_rate):
        self.learn_rate = learn_rate

    def update_model(self, sequential):
        for parameter in sequential.parameters():
            parameter.value -= self.learn_rate * parameter.gradient
    
            