from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from neura_command.network_utils.activation_functions import ActivationFunction
from neura_command.network_utils.loss_functions import DifferenceLoss, LossFunction


class NeuronInterface(ABC):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_func: ActivationFunction,
        loss_func: LossFunction = DifferenceLoss(),
        learning_rate: float = 0.1,
        l1_ratio: float = 0.0,
        l2_ratio: float = 0.0,
    ) -> None:
        pass

    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int):
        pass
