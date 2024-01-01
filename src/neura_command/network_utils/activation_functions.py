from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function to input x."""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the activation function with respect to input x."""
        pass


class LinearActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class SigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = self.compute(x)
        return sigmoid * (1 - sigmoid)


class BasicSigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)


class ReLUActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class StepActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class TanhActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
