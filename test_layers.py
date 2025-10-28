import unittest
import numpy as np
from dense import Dense
from activations import Sigmoid, Softmax

class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        self.input_size = 3
        self.output_size = 2
        self.dense_layer = Dense(self.input_size, self.output_size)

    def test_forward(self):
        input_data = np.array([[1.0], [2.0], [3.0]])
        output = self.dense_layer.forward(input_data)
        self.assertEqual(output.shape, (self.output_size, 1))

    def test_backward(self):
        input_data = np.array([[1.0], [2.0], [3.0]])
        output_gradient = np.array([[0.5], [0.5]])
        self.dense_layer.forward(input_data)
        input_gradient = self.dense_layer.backward(output_gradient, learning_rate=0.01)
        self.assertEqual(input_gradient.shape, (self.input_size, 1))

class TestActivationFunctions(unittest.TestCase):
    def setUp(self):
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()

    def test_sigmoid_forward(self):
        input_data = np.array([[0], [1], [-1]])
        output = self.sigmoid.forward(input_data)
        self.assertEqual(output.shape, input_data.shape)

    def test_softmax_forward(self):
        input_data = np.array([[1.0], [2.0], [3.0]])
        output = self.softmax.forward(input_data)
        self.assertAlmostEqual(np.sum(output), 1.0)

    def test_softmax_backward(self):
        input_data = np.array([[1.0], [2.0], [3.0]])
        output_gradient = np.array([[0.1], [0.2], [0.3]])
        self.softmax.forward(input_data)
        input_gradient = self.softmax.backward(output_gradient, learning_rate=0.01)
        self.assertEqual(input_gradient.shape, (3, 1))

if __name__ == '__main__':
    unittest.main()