import numpy as np
from typing import Callable

from neura_command.neuron.neuron_interface import NeuronInterface


class Perceptron(NeuronInterface):
    def __init__(
        self,
        input_size: int,
        activation_func: Callable[[np.ndarray], np.ndarray],
        activation_deriv: Callable[[np.ndarray], np.ndarray],
        error_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        learning_rate: float,
        l1_ratio: float,
        l2_ratio: float,
    ) -> None:
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.best_weights = np.copy(self.weights)
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        self.error_func = error_func
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.last_input = None
        self.gradient = None

    @property
    def input_size(self):
        return len(self.weights) - 1

    def forward(self, X: np.ndarray, add_bias: bool = True) -> np.ndarray:
        if add_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.last_input = X
        net_input = np.dot(X, self.weights)
        return self.activation_func(net_input)

    def backward(self, error: np.ndarray) -> np.ndarray:
        # Calculate derivative
        output = self.forward(self.last_input, add_bias=False)
        derivative = error * self.activation_deriv(output)

        # Ensure derivative is a 2D array for dot product
        if derivative.ndim == 1:
            derivative = derivative[:, np.newaxis]

        # Compute gradient with respect to weights
        self.gradient = (
            -np.dot(self.last_input.T, derivative) / self.last_input.shape[0]
        )

        # Calculate gradient with respect to the input
        # Reshape weights to 2D if necessary for the dot product
        if self.weights.ndim == 1:
            weights_reshaped = self.weights[:, np.newaxis]
        else:
            weights_reshaped = self.weights

        input_grad = np.dot(derivative, weights_reshaped.T)

        return input_grad

    def update_weights(self):
        # Ensure that self.gradient is a 1D array
        if self.gradient.ndim > 1:
            self.gradient = self.gradient.flatten()

        # Calculate L1 and L2 regularization terms
        l1_term = self.l1_ratio * np.sign(self.weights)
        l2_term = self.l2_ratio * self.weights

        # Ensure that l1_term and l2_term are 1D arrays
        if l1_term.ndim > 1:
            l1_term = l1_term.flatten()
        if l2_term.ndim > 1:
            l2_term = l2_term.flatten()

        # Update weights
        self.weights -= self.learning_rate * (self.gradient + l1_term + l2_term)

    def train(self, X, y, epochs, batch_size, verbose=False):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # Forward pass
                output = self.forward(X_batch, add_bias=True)

                # Compute error
                error = y_batch - output

                # Backward pass
                self.backward(error)

                # Update weights
                self.update_weights()

            if verbose and epoch % 100 == 0:
                current_error = self.error_func(y, self.forward(X))
                print(f"Epoch {epoch}, Error: {current_error}")
