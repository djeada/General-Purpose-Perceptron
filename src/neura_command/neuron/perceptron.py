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

    def _forward_pass(self, X_batch: np.ndarray) -> np.ndarray:
        # Ensure X_batch has the bias term and self.weights is correctly sized
        net_input = np.dot(X_batch, self.weights)  # X_batch should include the bias
        return self.activation_func(net_input)

    def _backward_pass(self, X_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        output = self._forward_pass(X_batch)
        error = y_batch - output

        # Ensure error and output are 2D arrays
        if error.ndim == 1:
            error = error[:, np.newaxis]
        if output.ndim == 1:
            output = output[:, np.newaxis]

        derivative = error * self.activation_deriv(output)

        # Compute gradient
        dE_dw = -np.dot(X_batch.T, derivative) / X_batch.shape[0]

        # Apply mean if applicable
        if dE_dw.ndim > 1:
            dE_dw = np.mean(dE_dw, axis=1, keepdims=True)

        return dE_dw.flatten()

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Perform a backward pass and calculate the gradient of the error
        with respect to the input. Store the gradient with respect to the weights.
        """
        # Calculate output
        output = self._forward_pass(self.last_input)

        # Calculate derivative
        derivative = error * self.activation_deriv(output)

        # Compute gradient with respect to weights
        self.gradient = (
            -np.dot(self.last_input.T, derivative) / self.last_input.shape[0]
        )

        # Calculate gradient with respect to the input
        input_grad = np.dot(derivative, self.weights.T)

        return input_grad

    def update_weights(self):
        """
        Update the weights using the stored gradients.
        """
        l1_term = self.l1_ratio * np.sign(self.weights)
        l2_term = self.l2_ratio * self.weights
        self.weights -= self.learning_rate * (self.gradient + l1_term + l2_term)

    def train(self, X, y, epochs, batch_size, verbose=False):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_bias[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # Forward pass
                output = self._forward_pass(X_batch)

                # Compute error
                error = y_batch - output

                # Backward pass
                self.backward(error)

                # Update weights
                self.update_weights()

            if verbose and epoch % 100 == 0:
                current_error = self.error_func(y, self._forward_pass(X_bias))
                print(f"Epoch {epoch}, Error: {current_error}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        net_input = np.dot(X_bias, self.weights)
        return self.activation_func(net_input)
