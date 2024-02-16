import numpy as np

from neura_command.neuron.perceptron.perceptron import Perceptron


class Layer:
    def __init__(self, input_size, num_perceptrons):
        self.perceptrons = [Perceptron(input_size) for _ in range(num_perceptrons)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.array([p.forward(X) for p in self.perceptrons]).T

    def backward(self, error: np.ndarray) -> np.ndarray:
        return np.sum([p.backward(error) for p in self.perceptrons], axis=0)

    def update_weights(self):
        for perceptron in self.perceptrons:
            perceptron.update_weights()
