import numpy as np
from typing import Callable

from neura_command.neuron.neuron_interface import NeuronInterface


class Perceptron(NeuronInterface):
    def __init__(
        self,
        input_size: int,
        output_size: int,  # Default to 1 for single output
        activation_func: Callable[[np.ndarray], np.ndarray],
        activation_deriv: Callable[[np.ndarray], np.ndarray],
        error_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        learning_rate: float,
        l1_ratio: float,
        l2_ratio: float,
    ) -> None:
        # Initialize weights as a matrix
        self.weights = np.random.normal(
            0, 1 / np.sqrt(input_size), (input_size + 1, output_size)
        )
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        self.error_func = error_func
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.last_input = None
        self.last_output = None
        self.gradient = None

    @property
    def input_size(self):
        return self.weights.shape[0] - 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Always add bias term
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.last_input = X
        net_input = np.dot(X, self.weights)
        self.last_output = self.activation_func(net_input)
        return self.last_output

    def backward(self, error: np.ndarray) -> np.ndarray:
        # Compute derivative using last_output since forward was already called
        derivative = error * self.activation_deriv(self.last_output)

        if derivative.ndim == 1:
            derivative = derivative[:, np.newaxis]

        self.gradient = -np.dot(self.last_input.T, derivative)
        input_grad = np.dot(derivative, self.weights.T)

        # Exclude the gradient with respect to the bias term
        return derivative, input_grad[:, 1:]

    def update_weights(self):
        l1_term = self.l1_ratio * np.sign(self.weights)
        l2_term = self.l2_ratio * self.weights

        self.weights -= self.learning_rate * (self.gradient + l1_term + l2_term)

    def train(self, X, y, epochs, batch_size, verbose=False):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                output = self.forward(X_batch)
                error = self.error_func(y_batch, output)

                self.backward(error)
                self.update_weights()
