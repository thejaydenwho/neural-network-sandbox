import numpy as np
import pandas as pd
import csv 

class Parameter:
    def __init__(self, value, gradient):
        self.value = value
        self.gradient = gradient

class Layer:
    def __init__(self):
        # Each layer must remember its inputs and outputs
        # during the forward pass to use them in the backward pass
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        # Takes input data, processes it, and returns the output
        # Must be overridden by the actual layer (Dense, ReLU, etc.)
        raise NotImplementedError("Forgot to implement forward pass")
    
    def backward(self, output_gradient):
        # Takes the gradient from layer ahead
        # Calculates partial derivatives and returns gradient for layer behind
        raise NotImplementedError("Forgot to implement backward pass")
    


class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []

    def add(self, layer):
        # Adds a new layer to the end of our neural network
        self.layers.append(layer)
    
    def forward(self, inputs, training=True):
        # Passes the data forward through every layer in order
        current_signal = inputs
        for layer in self.layers:
            current_signal = layer.forward(current_signal, training)
        return current_signal
    
    def backward(self, output_gradient):
        # Passes the output gradient backward through every layer in REVERSE order
        current_gradient = output_gradient
        for layer in reversed(self.layers):
            current_gradient = layer.backward(current_gradient)
        return current_gradient
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params

    def zero_gradients(self):
        for parameter in self.parameters():
            parameter.gradient = np.zeros_like(parameter.value)

class Trainer:
    def __init__(self, sequential, loss_function, optimizer):
        self.sequential = sequential
        self.loss_function = loss_function
        self.optimizer = optimizer

    def accuracy(self, y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        if y_pred.ndim == 2:
            preds = np.argmax(y_pred, axis=1)
            true = np.argmax(y_true, axis=1)
        else:
            preds = (y_pred > 0.5).astype(int).reshape(-1)
            true = y_true.reshape(-1)

        return np.sum(preds == true) / len(true)


    def train(self, X, y, epochs=10, batch_size=32, csv_file="training_log.csv"):
        csv_file = "TrainingData/"+csv_file
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss", "Accuracy"])
        for epoch in range(epochs):
            loss_sum = 0
            correct = 0
            total = 0

            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for start in range(0, len(X), batch_size):

                X_batch = X[start:start+batch_size]
                y_batch = y[start:start+batch_size]

                y_pred = self.sequential.forward(X_batch)

                loss = self.loss_function.forward(y_pred, y_batch)
                loss_sum += loss

                preds = np.argmax(y_pred, axis=1)
                true = np.argmax(y_batch, axis=1)

                correct += np.sum(preds == true)
                total += len(true)

                grad = self.loss_function.backward(y_pred, y_batch)

                self.sequential.backward(grad)
                self.optimizer.update_model(self.sequential)
                self.sequential.zero_gradients()
            
            epoch_loss = loss_sum / (len(X)//batch_size)
            epoch_accuracy = correct / total
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    epoch_loss,
                    epoch_accuracy
                ])
            print(f"Epoch {epoch} Loss: {epoch_loss}")
            print(f"Epoch {epoch} Accuracy: {epoch_accuracy}")
    
    def test(self, X, y):
        y_pred = self.sequential.forward(X, training=False)
        loss = self.loss_function.forward(y_pred, y)
        accuracy = self.accuracy(y_pred, y)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")
    
