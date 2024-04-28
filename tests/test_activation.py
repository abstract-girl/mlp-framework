import unittest
import numpy as np

from modules.activation import ReLU, Sigmoid, Tanh


class TestActivations(unittest.TestCase):
    def test_relu(self):
        relu = ReLU()
        input_array = np.array([-1, 0, 1, 2])
        expected_output = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(relu.forward(input_array), expected_output)

        grad_output = np.array([1, 1, 1, 1])
        expected_grad_input = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(relu.backward(grad_output), expected_grad_input)

    def test_sigmoid(self):
        sigmoid = Sigmoid()
        input_array = np.array([0, 2])
        expected_output = 1 / (1 + np.exp(-input_array))
        np.testing.assert_array_almost_equal(sigmoid.forward(input_array), expected_output)

        grad_output = np.array([1, 1])
        sigmoid_output = sigmoid.forward(input_array)
        expected_grad_input = sigmoid_output * (1 - sigmoid_output)
        np.testing.assert_array_almost_equal(sigmoid.backward(grad_output), expected_grad_input)

    def test_tanh(self):
        tanh = Tanh()
        input_array = np.array([-1, 0, 1])
        expected_output = np.tanh(input_array)
        np.testing.assert_array_almost_equal(tanh.forward(input_array), expected_output)

        grad_output = np.array([1, 1, 1])
        expected_grad_input = 1 - np.tanh(input_array) ** 2
        np.testing.assert_array_almost_equal(tanh.backward(grad_output), expected_grad_input)


if __name__ == '__main__':
    unittest.main()
