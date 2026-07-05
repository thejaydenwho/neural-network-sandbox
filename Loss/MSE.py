import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        n = y_true.size
        return (2 * (y_pred - y_true))/n

