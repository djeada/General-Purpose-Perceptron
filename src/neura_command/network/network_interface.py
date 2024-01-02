from abc import ABC, abstractmethod

import numpy as np

from neura_command.network_utils.loss_functions import LossFunction


class NetworkInterface(ABC):
    def __init__(self, loss_func: LossFunction):
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        pass

    @abstractmethod
    def update_weights(self) -> None:
        pass

    @abstractmethod
    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate_schedule: callable = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        pass
