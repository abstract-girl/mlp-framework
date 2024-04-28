from typing import List, Dict, Tuple
from abc import abstractmethod

import numpy as np
import pickle

from modules.activation import Activation
from modules.module import Module
from modules.mode import Mode


class Linear(Module):
    """
    Линейный слой (полносвязный слой) для использования в нейронных сетях.

    Атрибуты:
    - name (str): Имя слоя.
    - weights (np.array): Веса слоя, инициализируются случайными значениями.
    - bias (np.array): Смещения (биасы) слоя, инициализируются нулями.
    - grad_weights (np.array): Градиенты весов, используются при обновлении весов.
    - grad_bias (np.array): Градиенты смещений, используются при обновлении смещений.

    Методы:
    - forward(x): Прямой проход слоя.
    - backward(grad_output): Обратный проход слоя.
    - zero_grad(): Обнуление градиентов.
    - set_parameters(weights, bias): Установка весов и смещений.
    """

    def __init__(self, n_in: int, n_out: int, name: str):
        super().__init__()
        self.name = name
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2. / n_in)
        self.bias = np.zeros(n_out)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self._parameters = {
            f"{self.name}_weights": [self.weights, self.grad_weights],
            f"{self.name}_bias": [self.bias, self.grad_bias]
        }

    def forward(self, x):
        """
        Прямой проход через слой: вычисляет Wx + b.

        Параметры:
        - x (np.array): Входные данные.

        Возвращает:
        - Выходные данные слоя.
        """
        self.x = x  # Сохраняем вход для использования в backward
        return x.dot(self.weights) + self.bias

    def backward(self, grad_output):
        """
        Обратный проход: вычисляет градиенты по весам, смещениям и входу.

        Параметры:
        - grad_output (np.array): Градиент потерь по выходу слоя.

        Возвращает:
        - Градиент потерь по входу слоя.
        """
        self.grad_weights = self.x.T.dot(grad_output)
        self.grad_bias = grad_output.sum(axis=0)
        return grad_output.dot(self.weights.T)

    @property
    def parameters(self):
        """
        Возвращает параметры слоя и их градиенты.
        """
        self._parameters = {
            f"{self.name}_weights": [self.weights, self.grad_weights],
            f"{self.name}_bias": [self.bias, self.grad_bias]
        }
        return self._parameters

    def zero_grad(self):
        """
        Обнуляет градиенты параметров слоя.
        """
        self.grad_weights.fill(0)
        self.grad_bias.fill(0)

    def set_parameters(self, weights: np.array, bias: np.array):
        """
        Устанавливает параметры слоя.

        Параметры:
        - weights (np.array): Новые веса слоя.
        - bias (np.array): Новые смещения слоя.
        """
        self.weights = weights
        self.bias = bias


class MLP(Module):
    """
    Многослойный перцептрон (MLP), представляющий собой последовательность линейных слоев и активаций.

    Атрибуты:
    - layers (dict): Словарь, содержащий слои сети. Ключи - имена слоев, значения - экземпляры слоев.
    - _parameters (dict): Словарь параметров сети для оптимизации.

    Методы:
    - forward(x): Прямой проход через сеть.
    - backward(grad_output): Обратный проход через сеть.
    - zero_grad(): Обнуление градиентов всех параметров сети.
    - set_parameters(parameters): Установка параметров сети.
    """

    def __init__(self,
                 input_size: int,
                 mlp_hidden_config: List[Tuple[int, Activation]],
                 num_classes: int = None,
                 last_activation: Activation = None,
                 mode: Mode = Mode.CLASSIFICATION
                 ):
        """
        Инициализация многослойного перцептрона.

        Параметры:
        - input_size (int): Размер входного вектора.
        - mlp_hidden_config (List[Tuple[int, Activation]]): Конфигурация скрытых слоев, 
          содержащая пары (размер слоя, активационная функция).
        - num_classes (int, optional): Количество выходных классов. По умолчанию 1.
        - last_activation (Activation, optional): Активационная функция последнего слоя. По умолчанию None.
        """
        super().__init__()
        self.layers = dict()
        self.mode = mode

        if num_classes is None:
            num_classes = 1

        layer_sizes = [input_size] + [size for size, _ in mlp_hidden_config] + [num_classes]

        for i in range(len(layer_sizes) - 1):
            layer_name = f"layer_{i}"
            self.layers[layer_name] = Linear(layer_sizes[i], layer_sizes[i + 1], name=layer_name)
            if i < len(mlp_hidden_config):
                activation_name = f"activation_{i}"
                self.layers[activation_name] = mlp_hidden_config[i][1]()  # Добавляем экземпляр активационной функции

        if last_activation:
            activation_name = f"last_activation"
            self.layers[activation_name] = last_activation

        self._parameters = dict()

    def forward(self, x):
        """
        Прямой проход через сеть.

        Параметры:
        - x (np.array): Входные данные.

        Возвращает:
        - Выходные данные сети.
        """
        for layer in self.layers.values():
            x = layer(x)
        return x

    def backward(self, grad_output):
        """
        Обратный проход через сеть.

        Параметры:
        - grad_output (np.array): Градиент потерь по выходу сети.

        Возвращает:
        - Градиент потерь по входу сети.
        """
        for layer_key in reversed(self.layers.keys()):
            grad_output = self.layers[layer_key].backward(grad_output)
        return grad_output

    @property
    def parameters(self):
        """
        Возвращает параметры сети и их градиенты.
        """
        self._parameters = dict()
        for layer in self.layers.values():
            if isinstance(layer, Linear):  # Только для слоев с параметрами
                self._parameters.update(layer.parameters)
        return self._parameters

    def zero_grad(self):
        """
        Обнуляет градиенты всех параметров сети.
        """
        for layer_key in self.layers:
            if isinstance(self.layers[layer_key], Linear):
                self.layers[layer_key].zero_grad()

    def set_parameters(self, parameters):
        """
        Устанавливает параметры сети.

        Параметры:
        - parameters (dict): Словарь параметров.
        """
        for layer_key in self.layers:
            if isinstance(self.layers[layer_key], Linear):
                weights = parameters[f"{layer_key}_weights"][0]
                bias = parameters[f"{layer_key}_bias"][0]
                self.layers[layer_key].set_parameters(weights=weights, bias=bias)

    def __repr__(self):
        """
        Представление структуры многослойного перцептрона (MLP) в виде строки.
        
        Используется для удобного отображения архитектуры модели при печати экземпляра класса.
        Включает в себя информацию о типе и размерах каждого слоя.

        Возвращает:
        - model_architecture (str): Строковое представление архитектуры модели.
        """
        model_architecture = "MLP Architecture:\n"
        for layer_name, layer in self.layers.items():
            if isinstance(layer, Linear):
                layer_info = f"{layer_name}: Linear(in_features={layer.weights.shape[0]}, out_features={layer.weights.shape[1]})"
            elif isinstance(layer, Activation):  # Проверяем, является ли слой активацией
                layer_info = f"{layer_name}: {layer.__class__.__name__}"  # Получаем имя класса активационной функции
            else:
                layer_info = f"{layer_name}: Custom Layer"  # Для пользовательских слоев
            model_architecture += layer_info + "\n"
        return model_architecture

    def predict(self, x):
        """
        Генерирует предсказания на основе входных данных x.

        Параметры:
        - x (np.array): Входные данные.

        Возвращает:
        - predictions (np.array): Предсказания модели.
        """
        output = self.forward(x)
        if self.mode == Mode.CLASSIFICATION:
            # Для классификации возвращаем индекс максимального значения в выходных данных
            predictions = np.argmax(output, axis=1)
        elif self.mode == Mode.REGRESSION:
            # Для регрессии возвращаем непосредственно выходные данные
            predictions = output
        return predictions

    def save(self, path):
        """
        Сохраняет модель на диск.

        Параметры:
        - path (str): Путь, по которому будет сохранена модель.
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Модель успешно сохранена по пути: {path}")

    @staticmethod
    def load(path):
        """
        Загружает модель с диска.

        Параметры:
        - path (str): Путь, откуда будет загружена модель.

        Возвращает:
        - loaded_model (MLP): Экземпляр загруженной модели.
        """
        with open(path, 'rb') as file:
            loaded_model = pickle.load(file)
        print(f"Модель успешно загружена из файла: {path}")
        return loaded_model
