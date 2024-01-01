# Redefining the activation function and its derivative
from typing import List, Callable

import numpy as np

from neura_command.neuron.perceptron import Perceptron


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Redefining the loss function and its derivative for regression
def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


class FeedForwardNetwork:
    def __init__(
        self,
        layers: List[Perceptron],
        loss_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        loss_deriv: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):

        self.layers = layers

        self.loss_func = loss_func

        self.loss_deriv = loss_deriv

    def forward(self, X: np.ndarray) -> np.ndarray:

        output = X

        for layer in self.layers:

            output = layer.forward(output)

        return output

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):

        error = self.loss_deriv(y_pred, y_true)  # Flipping y_pred and y_true

        for layer in reversed(self.layers):

            delta, _ = layer.backward(error)

            # Update error for the next layer (exclude the bias term)

            error = delta @ layer.weights[:-1].T  # Exclude the bias weights

    def train(self, X, y, epochs, batch_size, X_val=None, y_val=None):

        for epoch in range(epochs):

            epoch_loss = 0

            for i in range(0, X.shape[0], batch_size):

                X_batch = X[i : i + batch_size]

                y_batch = y[i : i + batch_size]

                # Forward pass

                output = self.forward(X_batch)

                # Backward pass - calculate and propagate error

                self.backward(y_batch, output)

                # Update weights after each batch

                for layer in self.layers:

                    layer.update_weights()

                # Accumulate loss for the epoch

                epoch_loss += np.mean(self.loss_func(y_batch, output))

            # Average loss for the epoch

            epoch_loss /= X.shape[0] / batch_size

            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss}")

            # Validate if validation set is provided

            if X_val is not None and y_val is not None:

                val_loss = self.calculate_loss(X_val, y_val)

                print(f"Validation Loss: {val_loss}")

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:

        predictions = self.forward(X)

        return np.mean(self.loss_func(y, predictions))


# Regenerating the synthetic dataset for testing
np.random.seed(0)  # Ensuring reproducibility
X_synthetic = np.random.rand(100, 2)  # 100 samples, 2 features
y_synthetic = np.random.rand(100, 1)  # 100 samples, 1 target

# Redefining the input_size, hidden_layer_size, and output_size
input_size = X_synthetic.shape[1]  # Number of features in the dataset
hidden_layer_size = 5  # Number of neurons in the hidden layer
output_size = y_synthetic.shape[1]  # Number of output neurons (targets)

# Recreating the hidden and output layer
hidden_layer = Perceptron(
    input_size=input_size,
    output_size=hidden_layer_size,
    activation_func=sigmoid,
    activation_deriv=sigmoid_derivative,
    # learning_rate=3,
)

output_layer = Perceptron(
    input_size=hidden_layer_size,
    output_size=output_size,
    activation_func=sigmoid,
    activation_deriv=sigmoid_derivative,
    # learning_rate=2,
)

# Reinitializing and retraining the network with the revised FeedForwardNetwork class
network = FeedForwardNetwork(
    layers=[hidden_layer, output_layer],
    loss_func=mean_squared_error,
    loss_deriv=mse_derivative,
)

network.train(X_synthetic, y_synthetic, epochs=200, batch_size=10)
