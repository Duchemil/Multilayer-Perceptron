from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # better init: scaled gaussian + zero biases
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / max(1, input_size)) # He initialization, Weight Initialization 
        self.bias = np.zeros((output_size, 1))
        # velocity buffers for momentum
        self.vw = np.zeros_like(self.weights)
        self.vb = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input  # expected shape: (in_features, batch)
        return np.dot(self.weights, self.input) + self.bias  # (out_features, batch)

    def backward(self, output_gradient, learning_rate, momentum=0.0):
        # output_gradient shape: (out_features, batch)
        batch_size = output_gradient.shape[1]
        # gradients averaged over the batch:
        dw = (output_gradient @ self.input.T) / batch_size           # (out, in)
        db = np.sum(output_gradient, axis=1, keepdims=True) / batch_size  # (out, 1)

        if momentum > 0.0:
            # store previous velocities
            vw_prev = self.vw.copy()
            vb_prev = self.vb.copy()
            # update velocities
            self.vw = momentum * self.vw - learning_rate * dw
            self.vb = momentum * self.vb - learning_rate * db
            # Nesterov parameter update
            self.weights += -momentum * vw_prev + (1.0 + momentum) * self.vw
            self.bias += -momentum * vb_prev + (1.0 + momentum) * self.vb
        else:
            # Update parameters, no momentum, wikipedia for reference
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

        # gradient to pass to previous layer
        input_gradient = self.weights.T @ output_gradient  # (in, batch)
        return input_gradient