import numpy as np
import unittest
from modules.loss import MSE, MAE, CrossEntropy, BinaryCrossEntropy, HuberLoss, LogLikelihoodLoss


class TestLosses(unittest.TestCase):
    def test_mae_forward(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 4])
        loss = MAE()
        result = loss.forward(x, y)
        expected = 1 / 3
        self.assertAlmostEqual(result, expected)

    def test_mae_backward(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 4])
        loss = MAE()
        grad = loss.backward(x, y)
        expected_grad = np.array([0, 0, -1 / 3])
        np.testing.assert_array_almost_equal(grad, expected_grad)

    def test_mse_forward(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 4])
        loss = MSE()
        result = loss.forward(x, y)
        expected = 1 / 3
        self.assertAlmostEqual(result, expected)

    def test_mse_backward(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 4])
        loss = MSE()
        grad = loss.backward(x, y)
        expected_grad = np.array([0, 0, -2 / 3])
        np.testing.assert_array_almost_equal(grad, expected_grad)

    def test_cross_entropy_forward(self):
        x = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        y = np.array([[0, 0, 1], [1, 0, 0]])
        loss = CrossEntropy()
        result = loss.forward(x, np.log(y + 1e-15))
        self.assertTrue(result > 0)

    def test_cross_entropy_backward(self):
        x = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        y = np.array([[0, 0, 1], [1, 0, 0]])
        loss = CrossEntropy()
        grad = loss.backward(x, np.log(y + 1e-15))
        self.assertEqual(grad.shape, x.shape)

    def test_cross_entropy_forward(self):
        x = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
        y = np.array([[0, 0, 1], [1, 0, 0]])
        loss = CrossEntropy()
        result = loss.forward(x, y)
        self.assertTrue(result > 0)

    def test_binary_cross_entropy_backward(self):
        x = np.array([0.1, 0.9, 0.8])
        y = np.array([0, 1, 1])
        loss = BinaryCrossEntropy()
        grad = loss.backward(x, y)
        self.assertEqual(grad.shape, x.shape)

    def test_huber_loss_forward(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 4])
        loss = HuberLoss(delta=1.0)
        result = loss.forward(x, y)
        self.assertTrue(result > 0)

    def test_huber_loss_backward(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 4])
        loss = HuberLoss(delta=1.0)
        grad = loss.backward(x, y)
        self.assertEqual(grad.shape, x.shape)

    def test_log_likelihood_loss_forward(self):
        x = np.array([[2.0, 1.0], [1.0, 3.0]])
        y = np.array([0, 1])
        loss = LogLikelihoodLoss()
        result = loss.forward(x, y)
        probabilities = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        expected = -np.mean(np.log(probabilities[range(len(y)), y]))
        self.assertAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
