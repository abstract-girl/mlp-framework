from typing import List

from modules.datasets import Dataset
import numpy as np

from modules.transformer import Transformer


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False, feature_transformers: List[Transformer] = None,
                 target_transformers: List[Transformer] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_transformers = feature_transformers if feature_transformers is not None else []
        self.target_transformers = target_transformers if target_transformers is not None else []

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = list(range(len(self.dataset)))
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration
        indices = self.indices[self.idx:self.idx + self.batch_size]

        all_features = []
        all_targets = []

        for i in indices:
            all_features.append(self.dataset[i]['features'])
            all_targets.append(self.dataset[i]['targets'])

        # Apply each feature transformer in the chain
        for transformer in self.feature_transformers:
            all_features = [transformer.transform(feature) for feature in all_features]

        # Apply each target transformer in the chain
        for transformer in self.target_transformers:
            all_targets = [transformer.transform(target) for target in all_targets]

        all_features_array = np.vstack(all_features)
        all_targets_array = np.vstack(all_targets)

        self.idx += self.batch_size
        return {"features": all_features_array, "targets": all_targets_array}

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
