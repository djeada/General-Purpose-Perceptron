import random
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# Step 2: Define Loss Function and its Derivative
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)


def softmax(z: np.ndarray) -> np.ndarray:

    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Shift for numerical stability
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def softmax_derivative(softmax_output: np.ndarray, y_true: np.ndarray) -> np.ndarray:

    return softmax_output - y_true


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
        self.last_input = np.array([])
        self.last_output = np.array([])
        self.last_gradient = np.array([])

    @property
    def input_size(self) -> int:
        return self.weights.shape[0] - 1

    @property
    def output_size(self) -> int:
        return self.weights.shape[1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.last_input = np.c_[X, np.ones(X.shape[0])]
        z = self.last_input @ self.weights
        self.last_output = self.activation_func(z)
        return self.last_output

    def backward(self, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = error * self.activation_deriv(self.last_output)
        weight_gradient = -self.last_input.T @ delta
        self.last_gradient = weight_gradient

        return delta, weight_gradient

    def update_weights(self):
        l1_term = self.l1_ratio * np.sign(self.weights)
        l2_term = self.l2_ratio * self.weights

        self.weights -= self.learning_rate * (self.last_gradient + l1_term + l2_term)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int):
        for epoch in range(epochs):
            # Iterating through batches
            for i in range(0, X.shape[0], batch_size):
                # Slicing the batch
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Calculating error
                error = self.error_func(y_batch, output)

                # Backward pass and weight update
                self.backward(error)
                self.update_weights()


class FeedForwardNetwork:
    def __init__(self, layers, loss_func, loss_deriv):
        self.layers = layers
        self.loss_func = loss_func
        self.loss_deriv = loss_deriv

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true, y_pred):
        error = self.loss_deriv(y_pred, y_true)
        for layer in reversed(self.layers):
            error = layer.backward(error)[0]
            error = error @ layer.weights[:-1].T  # Exclude the bias weights

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()

    def train(
        self,
        X,
        y,
        epochs,
        batch_size,
        learning_rate_schedule=None,
        X_val=None,
        y_val=None,
    ):
        for epoch in range(epochs):
            # Apply learning rate schedule if provided
            if learning_rate_schedule:
                for layer in self.layers:
                    layer.learning_rate = learning_rate_schedule(epoch)

            # Training loop
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                output = self.forward(X_batch)
                self.backward(y_batch, output)
                self.update_weights()
                epoch_loss += np.mean(self.loss_func(y_batch, output))

            # Average loss for the epoch
            epoch_loss /= len(X) / batch_size
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss}")

            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self.calculate_loss(X_val, y_val)
                print(f"Validation Loss: {val_loss}")

    def calculate_loss(self, X, y):
        predictions = self.forward(X)
        return np.mean(self.loss_func(y, predictions))


def main():
    # Initialize constants and functions
    input_size = 784
    hidden_layer_size = 64
    output_size = 10

    # Create Perceptron layers
    layers = [
        Perceptron(
            input_size, hidden_layer_size, relu, relu_derivative, learning_rate=0.01
        ),
        Perceptron(
            hidden_layer_size,
            hidden_layer_size,
            relu,
            relu_derivative,
            learning_rate=0.01,
        ),
        Perceptron(
            hidden_layer_size, output_size, relu, relu_derivative, learning_rate=0.01
        ),
    ]

    # Create and train the network
    network = FeedForwardNetwork(layers, mse, mse_derivative)

    mnist = fetch_openml("mnist_784")
    X = mnist.data.astype("float32") / 255
    y = mnist.target.astype("int")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_labels = np.eye(10)[y_train]
    test_labels = np.eye(10)[y_test]

    network.train(X_train, train_labels, epochs=10, batch_size=10)
    test_loss = network.calculate_loss(X_test, test_labels)
    print(f"Test Loss: {test_loss}")

    # Step 1: Make predictions on the test set
    predicted_labels = np.argmax(network.forward(X_test), axis=1)

    # Assuming your test_labels are one-hot encoded, convert them to class labels
    true_labels = np.argmax(test_labels, axis=1)

    # Step 2: Compare predictions to true labels
    correct_predictions = np.sum(predicted_labels == true_labels)

    # Step 3: Calculate accuracy
    accuracy = (correct_predictions / len(true_labels)) * 100

    print(f"Accuracy on Test Set: {accuracy}%")

    # Convert X_test to a NumPy array if it's a DataFrame
    X_test_array = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

    # Define the number of samples to display
    num_samples = 10

    # Increase the overall figure size
    plt.figure(figsize=(15, 6))  # You can adjust the size as needed

    # Generate random indices to select samples
    sample_indices = random.sample(range(len(X_test_array)), num_samples)

    # Create a subplot for each sample in a 2x5 grid
    for i, index in enumerate(sample_indices):
        # Extract and reshape the image
        image = X_test_array[index].reshape(28, 28)

        # Get the true label
        true_label = np.argmax(test_labels[index])

        # Make a prediction using the network
        predicted_label = np.argmax(network.forward(X_test_array[index].reshape(1, -1)))

        # Create a subplot within a 2x5 grid
        plt.subplot(2, num_samples // 2, i + 1)

        # Display the image in grayscale
        plt.imshow(image, cmap="gray")

        # Set the title with predicted and true labels, adjust font size if needed
        plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", fontsize=10)

        # Turn off axis labels
        plt.axis("off")

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    plt.show()


if __name__ == "__main__":
    main()
