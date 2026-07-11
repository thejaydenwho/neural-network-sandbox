import numpy as np

class CrossEntropy: 
    def forward(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        eps = np.float32(1e-7)
        y_pred = np.clip(y_pred, eps, np.float32(1.0) - eps)
        row_sums = np.sum(y_true * np.log(y_pred), axis=1, dtype=np.float32)
        loss = -np.mean(row_sums, dtype=np.float32)
        return loss 

    def backward(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        batch_size = np.float32(y_true.shape[0])
        return (y_pred - y_true) / batch_size