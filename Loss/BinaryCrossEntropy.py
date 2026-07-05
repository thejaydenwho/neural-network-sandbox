import numpy as np

class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )
        return loss

    def backward(self, y_pred, y_true):
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size