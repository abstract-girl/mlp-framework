from abc import abstractmethod
import numpy as np
from modules.module import Module


class Activation(Module):
    """
    Абстрактный класс для активационных функций. Является подклассом Module.
    Предоставляет основу для реализации различных активационных функций.
    
    Методы:
    forward(x: np.array) -> np.array: Вычисляет прямое распространение активации.
                                      Должен быть переопределен в подклассах.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        """
        Абстрактный метод для прямого распространения. Вычисляет активацию входного тензора.
        Параметры:
            x (np.array): Входной тензор.
        Возвращает:
            np.array: Тензор после применения активационной функции.
        """
        output = None
        return output


class ReLU(Activation):
    """
    Класс ReLU (Rectified Linear Unit) активации.
    
    Методы:
    forward(x): Вычисляет прямое распространение активации ReLU.
    backward(grad_output): Вычисляет градиент функции потерь относительно входа.
    """

    def forward(self, x):
        """
        Вычисляет ReLU активацию.
        Параметры:
            x: Входные данные.
        Возвращает:
            Вывод активации ReLU.
        """
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad_output):
        """
        Вычисляет градиент функции потерь относительно входа для ReLU.
        Параметры:
            grad_output: Градиент потерь относительно выхода активации.
        Возвращает:
            Градиент потерь относительно входа активации.
        """
        grad_input = grad_output.copy()
        grad_input[self.output <= 0] = 0
        return grad_input


class Sigmoid(Activation):
    """
    Класс активации Sigmoid.
    
    Методы:
    forward(x): Вычисляет прямое распространение активации Sigmoid.
    backward(grad_output): Вычисляет градиент функции потерь относительно входа.
    """

    def forward(self, x):
        """
        Вычисляет активацию Sigmoid.
        Параметры:
            x: Входные данные.
        Возвращает:
            Вывод активации Sigmoid.
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        """
        Вычисляет градиент функции потерь относительно входа для Sigmoid.
        Параметры:
            grad_output: Градиент потерь относительно выхода активации.
        Возвращает:
            Градиент потерь относительно входа активации.
        """
        grad_input = grad_output * (self.output * (1 - self.output))
        return grad_input


class Softmax(Activation):
    """
    Класс активации Softmax.
    
    Методы:
    forward(x): Вычисляет прямое распространение активации Softmax.
    backward(grad_output): Вычисляет градиент функции потерь относительно входа.
    """

    def forward(self, x):
        """
        Вычисляет активацию Softmax.
        Параметры:
            x: Входные данные.
        Возвращает:
            Вывод активации Softmax.
        """
        # Сдвиг входа для численной стабильности
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        """
                Вычисляет градиент функции потерь относительно входа для Softmax.
        Параметры:
            grad_output: Градиент потерь относительно выхода активации.
        Возвращает:
            Градиент потерь относительно входа активации.
        """
        grad_input = np.empty_like(grad_output)
        for index, (single_output, single_grad_output) in enumerate(zip(self.output, grad_output)):
            # Переформирование для выполнения матричного умножения
            single_output = single_output.reshape(-1, 1)
            # Якобиан для функции Softmax
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Произведение якобиана функции Softmax и градиента потерь по выходу функции
            grad_input[index] = np.dot(jacobian_matrix, single_grad_output)
        return grad_input


class Tanh(Activation):
    """
    Класс активации Tanh (Гиперболический тангенс).
    
    Методы:
    forward(x): Вычисляет прямое распространение активации Tanh.
    backward(grad_output): Вычисляет градиент функции потерь относительно входа.
    """

    def forward(self, x):
        """
        Вычисляет активацию Tanh.
        Параметры:
            x: Входные данные.
        Возвращает:
            Вывод активации Tanh.
        """
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        """
        Вычисляет градиент функции потерь относительно входа для Tanh.
        Параметры:
            grad_output: Градиент потерь относительно выхода активации.
        Возвращает:
            Градиент потерь относительно входа активации.
        """
        grad_input = grad_output * (1 - self.output ** 2)
        return grad_input
