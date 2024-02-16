from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from neura_command.network_utils.activation_functions import BasicSigmoidActivation
from neura_command.neuron.perceptron.perceptron import Perceptron


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.square(y_true - y_pred))


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        self.layers = [
            Perceptron(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                learning_rate=learning_rate,
                activation_func=BasicSigmoidActivation(),
                loss_func=lambda a, b: a - b,
                l1_ratio=0,
                l2_ratio=0,
            )
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        activations = [X]
        for layer in self.layers:
            X = layer.forward(X)
            activations.append(X)
        return activations

    def backward(
        self, y: np.ndarray, activations: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        error = y - activations[-1]
        errors = [error]
        gradients = []
        for layer, activation in zip(reversed(self.layers), reversed(activations[:-1])):
            delta, weight_gradient = layer.backward(error)
            layer.last_gradient = weight_gradient
            layer.update_weights()
            error = np.dot(delta, layer.weights[:-1].T)
            errors.append(error)
            gradients.append(weight_gradient)
        return list(reversed(errors)), list(reversed(gradients))

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int
    ) -> Tuple[List[float], List[List[np.ndarray]], List[List[np.ndarray]]]:
        errors, weight_history, gradient_history = [], [], []
        for _ in range(epochs):
            activations = self.forward(X)
            errors.append(mean_squared_error(y, activations[-1]))
            _, gradients = self.backward(y, activations)
            weight_history.append([layer.weights for layer in self.layers])
            gradient_history.append(gradients)
        return errors, weight_history, gradient_history


def plot_training_error(errors: List[float]) -> None:
    plt.plot(errors, label="Training Error", color="blue")
    plt.title("Training Error over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()


def plot_gradient_evolution(
    gradient_history: List[List[List[float]]], nn: NeuralNetwork
) -> None:
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    for layer_index in range(len(nn.layers)):
        for neuron_index in range(nn.layers[layer_index].output_size):
            grads = [g[layer_index][neuron_index] for g in gradient_history]
            axes.plot(grads, label=f"Layer {layer_index + 1} Neuron {neuron_index + 1}")

    axes.set_title("Gradient Evolution for All Neurons")
    axes.set_xlabel("Epochs")
    axes.set_ylabel("Gradient Magnitude")
    axes.legend()
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(nn: NeuralNetwork, X: np.ndarray, y: np.ndarray) -> None:
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])[-1]
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.title("Decision Boundary for XOR problem")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


def main() -> None:
    # XOR input and output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Neural Network initialization and training
    nn = NeuralNetwork([2, 2, 1])
    errors, weight_history, gradient_history = nn.train(X, y, 20000)

    # Plotting error and gradient evolution
    plot_training_error(errors)
    plot_gradient_evolution(gradient_history, nn)
    plot_decision_boundary(nn, X, y)


if __name__ == "__main__":
    main()
