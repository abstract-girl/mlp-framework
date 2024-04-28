import unittest
import numpy as np

from modules.nn import MLP
from modules.activation import ReLU, Sigmoid


class TestMLP(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)  # Для воспроизводимости результатов
        self.input_size = 5
        self.num_classes = 3
        self.mlp = MLP(input_size=self.input_size,
                       mlp_hidden_config=[(10, ReLU), (7, Sigmoid)],
                       num_classes=self.num_classes)

    def test_forward_backward(self):
        # Простая проверка прямого прохода
        x = np.random.randn(1, self.input_size)
        output = self.mlp.forward(x)
        self.assertEqual(output.shape, (1, self.num_classes))

        # Проверка обратного прохода и обновления параметров
        output_grad = np.random.randn(1, self.num_classes)
        self.mlp.backward(output_grad)

        # Простая проверка на изменение весов после обратного прохода
        # Здесь мы не ожидаем конкретных значений, только убеждаемся, что градиенты не нулевые
        for params in self.mlp.parameters:
            weight, grad = params
            self.assertFalse(np.allclose(grad, 0), "Gradient should not be zero")


if __name__ == '__main__':
    unittest.main()
