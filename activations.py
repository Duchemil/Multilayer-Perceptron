from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate, momentum=0.0):
        return np.multiply(output_gradient, self.activation_prime(self.input)) # return dL/dX
    
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x) # ReLU function
        
        def relu_prime(x):
            return (x > 0).astype(float) # Derivative of ReLU
        
        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x)) # Sigmoid function
        
        def sigmoid_prime(x):
            s = sigmoid(x) 
            return s * (1 - s) # Derivative of Sigmoid
        
        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        # input shape: (n_classes, batch)
        exps = np.exp(input - np.max(input, axis=0, keepdims=True))
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate, momentum=0.0):
        # output_gradient shape: (n_classes, batch)
        s = self.output
        # vectorized Jacobian-vector product per sample
        return s * (output_gradient - np.sum(output_gradient * s, axis=0, keepdims=True))
