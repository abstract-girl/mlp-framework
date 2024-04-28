from abc import ABC, abstractmethod
import numpy as np


class Transformer(ABC):

    @abstractmethod
    def transform(self, data):
        """Обработка данных"""
        pass


class NormalizeTransformer(Transformer):
    def transform(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


class MinMaxScalerTransformer(Transformer):
    def transform(self, data):
        return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


class OneHotEncodingTransformer(Transformer):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def transform(self, data):
        # Initialize the one-hot encoded array with zeros
        one_hot_encoded_data = np.zeros((data.size, self.num_classes))
        # np.arange(data.size) creates an array of indices;
        # data.ravel() flattens the data array if it's multidimensional
        one_hot_encoded_data[np.arange(data.size), data.ravel()] = 1
        return one_hot_encoded_data
