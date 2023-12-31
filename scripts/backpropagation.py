import numpy as np
import matplotlib.pyplot as plt
from neura_command.neuron.perceptron import Perceptron
from typing import List, Tuple

# Activation function and its derivative
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output: np.ndarray) -> np.ndarray:
    return sigmoid_output * (1 - sigmoid_output)


# Error function
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.square(y_true - y_pred))


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        self.layers = [
            Perceptron(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                learning_rate=learning_rate,
                activation_func=sigmoid,
                activation_deriv=sigmoid_derivative,
                error_func=lambda a, b: a - b,
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


def main() -> None:
    # XOR input and output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Neural Network initialization and training
    nn = NeuralNetwork([2, 2, 1], learning_rate=0.1)
    errors, weight_history, gradient_history = nn.train(X, y, 20000)

    # Plotting error and gradient evolution
    plot_training_error(errors)
    plot_gradient_evolution(gradient_history, nn)


if __name__ == "__main__":
    main()
