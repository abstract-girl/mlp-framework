from abc import abstractmethod, ABC
from typing import Dict, List

import numpy as np


class Module(ABC):
    # List[np.ndarray <- current weight, np.ndarray <- grad of weight]
    parameters: Dict[str, List[np.ndarray]] = dict()

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *grad_previous_output):
        # Compute gradients (backward pass)
        raise NotImplementedError

    @property
    def parameters(self):
        """Return parameters and their gradients. Default implementation for layers without parameters."""
        return {}

    def zero_grad(self):
        """Reset gradients of all parameters to zero. Default implementation for layers without parameters."""
        pass

    def __call__(self, *inputs):
        return self.forward(*inputs)
