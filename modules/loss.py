from abc import abstractmethod

import numpy as np
from modules.module import Module


class Loss(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x: np.array, y: np.array) -> float:
        pass

    @abstractmethod
    def backward(self, x: np.array, y: np.array) -> float:
        pass


class MAE(Loss):
    def forward(self, x: np.array, y: np.array) -> float:
        try:
            return np.mean(np.abs(x - y))
        except Exception as e:
            raise ValueError(f"Ошибка MAE forward: {e}")

    def backward(self, x: np.array, y: np.array):
        try:
            return np.sign(x - y) / y.size
        except Exception as e:
            raise ValueError(f"Ошибка MAE backward: {e}")


class MSE(Loss):
    def forward(self, x: np.array, y: np.array) -> float:
        try:
            return np.mean(np.square(x - y))
        except Exception as e:
            raise ValueError(f"Ошибка MSE forward: {e}")

    def backward(self, x: np.array, y: np.array):
        try:
            return 2 * (x - y) / y.size
        except Exception as e:
            raise ValueError(f"Ошибка MSE backward: {e}")


class CrossEntropy(Loss):
    def forward(self, x: np.array, y: np.array) -> float:
        clipped_x = np.clip(x, 1e-12, 1 - 1e-12)
        loss = -np.sum(y * np.log(clipped_x)) / x.shape[0]
        return loss

    def backward(self, x: np.array, y: np.array) -> np.array:
        clipped_x = np.clip(x, 1e-12, 1 - 1e-12)
        grad = -y / clipped_x
        grad /= x.shape[0]
        return grad


class BinaryCrossEntropy(Loss):
    def forward(self, x: np.array, y: np.array) -> float:
        try:
            return -np.mean(y * np.log(x + 1e-15) + (1 - y) * np.log(1 - x + 1e-15))
        except Exception as e:
            raise ValueError(f"Ошибка BinaryCrossEntropy forward: {e}")

    def backward(self, x: np.array, y: np.array):
        try:
            return (x - y) / (x * (1 - x) + 1e-15)
        except Exception as e:
            raise ValueError(f"Ошибка BinaryCrossEntropy backward: {e}")


class HuberLoss(Loss):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, x: np.array, y: np.array) -> float:
        try:
            diff = np.abs(x - y)
            return np.mean(np.where(diff <= self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta)))
        except Exception as e:
            raise ValueError(f"Ошибка HuberLoss forward: {e}")

    def backward(self, x: np.array, y: np.array):
        try:
            diff = x - y
            return np.where(np.abs(diff) <= self.delta, diff, self.delta * np.sign(diff))
        except Exception as e:
            raise ValueError(f"Ошибка HuberLoss backward: {e}")


class LogLikelihoodLoss(Module):
    def forward(self, x: np.array, y: np.array) -> float:
        try:
            probabilities = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
            log_likelihood = -np.log(probabilities[range(len(y)), y])
            return np.mean(log_likelihood)
        except Exception as e:
            raise ValueError(f"Ошибка LogLikelihoodLoss forward: {e}")

    def backward(self, x: np.array, y: np.array):
        try:
            probabilities = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
            probabilities[range(len(y)), y] -= 1
            return probabilities / len(y)
        except Exception as e:
            raise ValueError(f"Ошибка LogLikelihoodLoss backward: {e}")
