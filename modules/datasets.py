from abc import abstractmethod

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_openml


class Dataset:
    """
    Only original objects
    """

    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Get the sample at index `idx`."""
        pass


class MnistDataset(Dataset):
    def __init__(self):
        print("Loading MNIST dataset...")
        self.mnist_data = fetch_openml('mnist_784', version=1, cache=True)
        print("Done!")
        self.features = self.mnist_data.data.to_numpy()
        self.target = self.mnist_data.target.to_numpy().astype(int)
        assert self.features.shape[0] == self.target.shape[
            0], "The number of images does not match the number of labels."
        self.labels = [str(i) for i in range(10)]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'targets': np.array(self.target[idx])}


class IrisDataset(Dataset):
    def __init__(self):
        self.ds_iris = datasets.load_iris()
        self.features = self.ds_iris.data
        self.target = self.ds_iris.target
        assert self.features.shape[0] == self.target.shape[
            0], f"self.features.shape[0] == self.target.shape[0]: {self.features.shape[0]} != {self.target.shape[0]}"
        self.labels = self.ds_iris.target_names

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'targets': np.array(self.target[idx])}


class DiabetesDataset(Dataset):
    def __init__(self):
        self.ds_diabetes = load_diabetes()
        self.features = self.ds_diabetes.data
        self.target = self.ds_diabetes.target
        assert self.features.shape[0] == self.target.shape[0], "Mismatch in number of features and targets."

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'targets': np.array(self.target[idx])}


class NumberDataset(Dataset):
    def __init__(self, start, end):
        self.numbers = list(range(start, end + 1))

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        return {'number': self.numbers[idx]}
