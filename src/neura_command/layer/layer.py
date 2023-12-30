import numpy as np
from typing import List
from neura_command.neuron.neuron_interface import NeuronInterface


class Layer:
    def __init__(self, neurons: List[NeuronInterface]) -> None:
        self.neurons = neurons

    def forward(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a NumPy array.")

        if len(self.neurons) == 0:
            raise ValueError("The layer contains no neurons.")

        # Concatenate the outputs from each neuron along the second dimension
        outputs = np.hstack(
            [neuron.forward(X)[:, np.newaxis] for neuron in self.neurons]
        )
        return outputs

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer. Computes and propagates the gradient of the error.
        Returns the gradient of the error with respect to the layer's inputs.
        """
        input_grad = np.zeros_like(error)
        for neuron, neuron_error in zip(self.neurons, error.T):
            # Assuming each neuron has a backward method that calculates the gradient
            # of the error with respect to its input and updates its internal state
            # for the weight gradient.
            grad = neuron.backward(neuron_error)
            input_grad += grad

        return input_grad

    def update_weights(self, learning_rate: float) -> None:
        """
        Update the weights of each neuron in the layer.
        """
        for neuron in self.neurons:
            # Assuming each neuron has an update_weights method
            neuron.update_weights(learning_rate)
