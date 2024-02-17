import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from neura_command.layer.layer import Layer
from neura_command.network.feed_forward_network.feed_forward_network import (
    FeedForwardNetwork,
)
from neura_command.network_utils.activation_functions import ReLUActivation
from neura_command.network_utils.loss_functions import MSELoss
from neura_command.neuron.perceptron.perceptron import Perceptron


def initialize_network(
    input_size: int, hidden_layer_size: int, output_size: int, learning_rate: float
) -> FeedForwardNetwork:
    """
    Initializes the neural network with given parameters.
    """
    layers = [
        Layer(
            [
                Perceptron(
                    input_size,
                    hidden_layer_size,
                    activation_func=ReLUActivation(),
                    learning_rate=learning_rate,
                )
            ]
        ),
        Layer(
            [
                Perceptron(
                    hidden_layer_size,
                    hidden_layer_size,
                    activation_func=ReLUActivation(),
                    learning_rate=learning_rate,
                )
            ]
        ),
        Layer(
            [
                Perceptron(
                    hidden_layer_size,
                    output_size,
                    activation_func=ReLUActivation(),
                    learning_rate=learning_rate,
                )
            ]
        ),
    ]

    network = FeedForwardNetwork(loss_func=MSELoss())
    for layer in layers:
        network.add_layer(layer)

    return network


def load_and_prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the MNIST dataset.
    """
    mnist = fetch_openml("mnist_784")
    X = mnist.data.astype("float32") / 255
    X = X.to_numpy()  # Convert to NumPy array if it's a DataFrame
    y = mnist.target.astype("int")
    y = y.to_numpy()  # Convert to NumPy array if it's a DataFrame or Series
    return train_test_split(X, y, test_size=0.2, random_state=42)


def calculate_accuracy(
    network: FeedForwardNetwork, X_test: np.ndarray, test_labels: np.ndarray
) -> float:
    """
    Calculate and return the accuracy of the network.
    """
    predicted_labels = np.argmax(network.forward(X_test), axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    correct_predictions = np.sum(predicted_labels == true_labels)
    return (correct_predictions / len(true_labels)) * 100


def display_sample_predictions(
    X_test: np.ndarray,
    test_labels: np.ndarray,
    network: FeedForwardNetwork,
    num_samples: int = 10,
) -> None:
    """
    Display sample predictions from the network.
    """
    sample_indices = random.sample(range(len(X_test)), num_samples)

    plt.figure(figsize=(15, 6))
    for i, index in enumerate(sample_indices):
        image = X_test[index].reshape(28, 28)  # Ensure X_test is a NumPy array
        true_label = np.argmax(test_labels[index])
        predicted_label = np.argmax(network.forward(X_test[index].reshape(1, -1)))
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", fontsize=10)
        plt.axis("off")
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()


def main() -> None:
    input_size, hidden_layer_size, output_size, learning_rate = 784, 64, 10, 0.01
    network = initialize_network(
        input_size, hidden_layer_size, output_size, learning_rate
    )
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    train_labels = np.eye(10)[y_train]
    test_labels = np.eye(10)[y_test]

    network.train(X_train, train_labels, epochs=2, batch_size=10)
    test_loss = network.calculate_loss(X_test, test_labels)
    print(f"Test Loss: {test_loss}")
    accuracy = calculate_accuracy(network, X_test, test_labels)
    print(f"Accuracy on Test Set: {accuracy}%")
    display_sample_predictions(X_test, test_labels, network)


if __name__ == "__main__":
    main()
