from typing import Dict, Tuple

import numpy as np


class Optimizer:

    # Метод step для выполнения шага оптимизации, должен быть переопределен в подклассах
    def step(self, parameters):
        raise NotImplementedError

    # Обнуление градиентов после обновления весов
    def zero_grad(self, parameters):
        for param_name in parameters.keys():
            parameters[param_name][1].fill(0)
        return parameters


class SGD(Optimizer):
    # Конструктор класса SGD, принимает параметры и learning rate (lr)
    def __init__(self, lr=0.01):
        self.lr = lr

    # Метод step для выполнения шага SGD оптимизации
    def step(self, parameters):
        for param_name, (weight, grad) in parameters.items():
            # Обновление веса по формуле градиентного спуска
            parameters[param_name][0] -= self.lr * grad
        return parameters


class MomentumSGD(Optimizer):
    # Класс MomentumSGD наследуется от Optimizer и реализует метод импульсного стохастического градиентного спуска

    def __init__(self, lr=0.01, momentum=0.9):
        # Конструктор класса, принимает параметры, learning rate (lr) и коэффициент импульса (momentum)       
        self.lr = lr  # Присвоение learning rate
        self.momentum = momentum  # Присвоение коэффициента импульса
        self.velocities = {}  # Словарь для хранения скоростей импульса для каждого параметра

    def step(self, parameters):
        for param_name, (weight, grad) in parameters.items():
            if param_name not in self.velocities:
                self.velocities[param_name] = np.zeros_like(weight)
            self.velocities[param_name] = self.momentum * self.velocities[param_name] - self.lr * grad
            parameters[param_name][0] += self.velocities[param_name]
        return parameters


class GradientClipping(Optimizer):
    # Класс GradientClipping наследуется от Optimizer и реализует метод обрезки градиентов

    def __init__(self, lr=0.01, momentum=0.9, max_norm=1.0):
        # Конструктор класса, принимает learning rate (lr), коэффициент импульса (momentum) и максимальную норму для обрезки градиентов
        self.lr = lr  # Присвоение learning rate
        self.momentum = momentum  # Присвоение коэффициента импульса
        self.max_norm = max_norm  # Присвоение максимальной нормы
        self.velocities = {}  # Словарь для хранения скоростей импульса для каждого параметра

    def step(self, parameters):
        total_norm = 0

        # Вычисление общей нормы градиентов
        for param_name, (_, grad) in parameters.items():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        # Вычисление коэффициента обрезки
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param_name in parameters:
                parameters[param_name][1] *= clip_coef

        # Применение обновлений с учетом возможной обрезки градиентов
        for param_name, (weight, grad) in parameters.items():
            if param_name not in self.velocities:
                self.velocities[param_name] = np.zeros_like(weight)
            # Обновление скорости с учетом обрезанных градиентов
            self.velocities[param_name] = self.momentum * self.velocities[param_name] - self.lr * grad
            # Обновление весов
            weight += self.velocities[param_name]
            parameters[param_name][0] = weight

        return parameters


class Adam(Optimizer):
    # Класс Adam наследуется от Optimizer и реализует метод оптимизации Adam

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Конструктор класса, принимает learning rate (lr), параметры beta1, beta2 и epsilon
        self.lr = lr  # Присвоение learning rate
        self.beta1 = beta1  # Присвоение параметра beta1
        self.beta2 = beta2  # Присвоение параметра beta2
        self.epsilon = epsilon  # Присвоение параметра epsilon
        self.m = {}  # Словарь для хранения первого момента
        self.v = {}  # Словарь для хранения второго момента
        self.t = 0  # Инициализация шага оптимизации

    def step(self, parameters):
        # Метод step для выполнения шага оптимизации методом Adam

        self.t += 1  # Увеличение шага оптимизации

        for param_name, (weight, grad) in parameters.items():
            # Итерация по параметрам и их градиентам

            if param_name not in self.m:
                # Если первый момент для данного параметра еще не существует
                self.m[param_name] = np.zeros_like(weight)  # Инициализация нулевым массивом
                self.v[param_name] = np.zeros_like(weight)  # Инициализация нулевым массивом

            # Обновление первого и второго моментов по формулам Adam
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)  # Исправление смещения в первом моменте
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)  # Исправление смещения во втором моменте

            # Обновление параметра модели по формуле Adam
            parameters[param_name][0] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameters


class AdaGrad(Optimizer):
    # Класс AdaGrad наследуется от Optimizer и реализует метод оптимизации AdaGrad

    def __init__(self, lr=0.01, epsilon=1e-8):
        # Конструктор класса, принимает learning rate (lr) и epsilon
        self.lr = lr  # Присвоение learning rate
        self.epsilon = epsilon  # Присвоение epsilon
        self.G = {}  # Словарь для хранения суммы квадратов градиентов

    def step(self, parameters):
        # Метод step для выполнения шага оптимизации методом AdaGrad

        for param_name, (weight, grad) in parameters.items():
            # Итерация по параметрам и их градиентам

            if param_name not in self.G:
                # Если сумма квадратов градиентов для данного параметра еще не существует
                self.G[param_name] = np.zeros_like(weight)  # Инициализация нулевым массивом

            # Обновление суммы квадратов градиентов
            self.G[param_name] += grad ** 2

            # Обновление параметра модели по формуле AdaGrad
            parameters[param_name][0] -= (self.lr / (np.sqrt(self.G[param_name]) + self.epsilon)) * grad

        return parameters
