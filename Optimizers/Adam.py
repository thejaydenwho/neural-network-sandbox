import numpy as np
from Base import *
class Adam:
    def __init__(self, learn_rate=0.001, b1=0.9, b2=0.999, epsilon=1e-8):
        self.learn_rate = np.float32(learn_rate)
        self.b1 = np.float32(b1)
        self.b2 = np.float32(b2)
        self.epsilon = np.float32(epsilon)
        # moment 1 and 2
        self.m = {}
        self.v = {}
        # turns done
        self.t = 0

    def update_model(self, sequential):
        self.t += 1
        for parameter in sequential.parameters():
            grad = parameter.gradient.astype(np.float32)
            key = id(parameter)
            if key not in self.m:
                self.m[key] = np.zeros(parameter.gradient.shape, dtype=np.float32)
                self.v[key] = np.zeros(parameter.gradient.shape, dtype=np.float32)
            self.m[key] = ((self.b1 * self.m[key]) + ((np.float32(1.0) - self.b1) * grad))
            self.v[key] = ((self.b2 * self.v[key]) + ((np.float32(1.0) - self.b2)) * (grad ** 2))
            mhat = self.m[key] / (np.float32(1.0) - (self.b1 ** self.t))
            vhat = self.v[key] / (np.float32(1.0) - (self.b2 ** self.t))
            update = (self.learn_rate * mhat/(np.sqrt(vhat) + self.epsilon)).astype(np.float32)

            parameter.value -= update
            

            
            

            



    