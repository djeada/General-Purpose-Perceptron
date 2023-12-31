import numpy as np
import matplotlib.pyplot as plt


# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)  # This is correct as x is the output of sigmoid


# Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


import numpy as np
from typing import Callable


class Perceptron:
    def __init__(
            self,
            input_size,
            output_size,
            activation_func: Callable[[np.ndarray], np.ndarray],
            activation_deriv: Callable[[np.ndarray], np.ndarray],
            learning_rate=0.1,
            error_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda a, b: a - b,
            l1_ratio: float = 0.0,
            l2_ratio: float = 0.0
    ):
        self.weights = np.random.normal(
            0, 1 / np.sqrt(input_size), (input_size + 1, output_size)
        )
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        self.error_func = error_func

        # Storing the last input and output
        self.last_input = None
        self.last_output = None

    @property
    def input_size(self) -> int:
        """Size of the input layer."""
        return self.weights.shape[0] - 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        bias_column = np.ones((X.shape[0], 1))
        self.last_input = np.hstack((X, bias_column))
        self.last_output = self.activation_func(
            np.dot(self.last_input, self.weights))
        return self.last_output

    def backward(self, error):
        delta = error * self.activation_deriv(self.last_output)
        weight_gradient = np.dot(self.last_input.T, delta)
        return delta, weight_gradient

    def update_weights(self, weight_gradient):
        self.weights += weight_gradient * self.learning_rate

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 1):
        """Train the network for a fixed number of epochs."""
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i: i + batch_size]
                y_batch = y[i: i + batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Compute error
                error = self.error_func(y_batch, output)

                # Backward pass and update weights
                _, weight_gradient = self.backward(error)
                self.update_weights(weight_gradient)



class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Perceptron(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    learning_rate=learning_rate,
                    activation_func=sigmoid,
                    activation_deriv=sigmoid_derivative,
                )
            )

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y):
        error = y - self.layers[-1].last_output
        for layer in reversed(self.layers):
            delta, weight_gradient = layer.backward(error)
            layer.update_weights(weight_gradient)
            # Exclude the bias term from the weights during error calculation
            error = np.dot(delta, layer.weights[:-1].T)

    def train(self, X, y, epochs):
        errors = []
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(y)
            errors.append(self.mean_squared_error(y, output))
        return errors

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()


# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Neural Network initialization and training
nn = NeuralNetwork([2, 2, 1], learning_rate=0.1)
errors = nn.train(X, y, 20000)
print(min(errors))
# Plotting the error over epochs
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Error over Epochs with Adjusted Network Architecture")
plt.show()
