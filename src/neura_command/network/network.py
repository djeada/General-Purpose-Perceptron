from typing import Callable, List
import numpy as np

from src.neura_command.layer.layer import Layer


class Network:
    def __init__(
        self,
        loss_func: Callable[[np.ndarray, np.ndarray], float],
        error_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.layers = []
        self.loss_func = loss_func
        self.error_func = error_func

    def add_layer(self, layer: Layer):
        if self.layers and self.layers[-1].output_size != layer.input_size:
            raise ValueError("Layer sizes are not compatible.")
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, error: np.ndarray):
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def update_weights(self, learning_rate: float):
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        verbose: bool = False,
    ):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i : i + batch_size]
                Y_batch = Y[i : i + batch_size]

                output = self.forward(X_batch)
                error = self.error_func(Y_batch, output)
                self.backward(error)
                self.update_weights(learning_rate)

            if verbose and epoch % 100 == 0:
                loss = self.loss_func(Y, self.forward(X))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
