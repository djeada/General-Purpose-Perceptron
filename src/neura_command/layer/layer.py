from typing import List

import numpy as np

from neura_command.neuron.neuron_interface import NeuronInterface


class Layer:
    def __init__(self, neurons: List[NeuronInterface]) -> None:
        if not neurons:
            raise ValueError("Layer must have at least one neuron.")

        self.neurons = neurons

    @property
    def input_size(self) -> int:
        return self.neurons[0].input_size

    @property
    def output_size(self) -> int:
        return len(self.neurons)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a NumPy array.")

        # Concatenate the outputs from each neuron along the second dimension
        outputs = np.hstack(
            [neuron.forward(X)[:, np.newaxis] for neuron in self.neurons]
        )
        return outputs

    def backward(self, error: np.ndarray) -> np.ndarray:
        input_grad = np.zeros_like(error)
        for neuron, neuron_error in zip(self.neurons, error.T):
            grad = neuron.backward(neuron_error)
            input_grad += grad

        return input_grad

    def update_weights(self, learning_rate: float) -> None:
        for neuron in self.neurons:
            neuron.learning_rate = learning_rate
            neuron.update_weights()

    def __len__(self) -> int:
        """
        Return the number of neurons in the layer.
        """
        return len(self.neurons)
