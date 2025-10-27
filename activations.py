from activation import Activation
from layer import Layer
import numpy as np

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
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp) 
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        