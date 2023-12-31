import numpy as np
from typing import Callable


class Perceptron:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_func: Callable[[np.ndarray], np.ndarray],
        activation_deriv: Callable[[np.ndarray], np.ndarray],
        learning_rate: float = 0.1,
        error_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda a, b: a - b,
        l1_ratio: float = 0.0,
        l2_ratio: float = 0.0,
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
        self.last_gradient = None

    @property
    def input_size(self) -> int:
        return self.weights.shape[0] - 1

    @property
    def output_size(self) -> int:
        return self.weights.shape[1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        bias_column = np.ones((X.shape[0], 1))
        self.last_input = np.hstack((X, bias_column))
        self.last_output = self.activation_func(np.dot(self.last_input, self.weights))
        return self.last_output

    def backward(self, error: np.ndarray) -> np.ndarray:
        delta = error * self.activation_deriv(self.last_output)
        weight_gradient = -np.dot(self.last_input.T, delta)
        self.last_gradient = weight_gradient
        return delta, weight_gradient

    def update_weights(self):
        l1_term = self.l1_ratio * np.sign(self.weights)
        l2_term = self.l2_ratio * self.weights

        self.weights -= self.learning_rate * (self.last_gradient + l1_term + l2_term)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                output = self.forward(X_batch)
                error = self.error_func(y_batch, output)

                self.backward(error)
                self.update_weights()
