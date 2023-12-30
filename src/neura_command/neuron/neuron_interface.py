import numpy as np
from abc import ABC, abstractmethod


class NeuronInterface(ABC):
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, verbose: bool
    ):
        pass
