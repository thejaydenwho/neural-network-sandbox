import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        return np.mean((y_pred - y_true) ** 2, dtype=np.float32)

    def backward(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        n = np.float32(y_true.size)
        return (np.float32(2.0) * (y_pred - y_true)) / n


