from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # better init: scaled gaussian + zero biases
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / max(1, input_size)) # He initialization, Weight Initialization 
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input  # expected shape: (in_features, batch)
        return np.dot(self.weights, self.input) + self.bias  # (out_features, batch)

    def backward(self, output_gradient, learning_rate):
        # output_gradient shape: (out_features, batch)
        batch_size = output_gradient.shape[1]
        # gradients averaged over the batch:
        dw = (output_gradient @ self.input.T) / batch_size           # (out, in)
        db = np.sum(output_gradient, axis=1, keepdims=True) / batch_size  # (out, 1)
        # update parameters
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        # gradient to pass to previous layer
        input_gradient = self.weights.T @ output_gradient  # (in, batch)
        return input_gradient