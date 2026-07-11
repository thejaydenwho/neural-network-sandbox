import numpy as np

class BinaryCrossEntropy: 
    def forward(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        epsilon = np.float32(1e-7)
        y_pred = np.clip(y_pred, epsilon, np.float32(1.0) - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred),
            dtype=np.float32
        )
        return loss 

    def backward(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        batch_size = np.float32(y_true.shape[0])
        return (y_pred - y_true) / batch_size