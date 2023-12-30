import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from neura_command.neuron.perceptron import Perceptron

# Activation functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


# Error function
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


# Function to plot decision boundary
def plot_decision_boundary(
    perceptron: Perceptron,
    X: np.ndarray,
    y: np.ndarray,
    iteration: int,
    weights: np.ndarray,
    ax: plt.Axes,
) -> None:
    # Set the perceptron's weights to the saved state
    perceptron.weights = weights
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = perceptron.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    ax.set_title(f"Decision Boundary at Epoch {iteration}")


# Function to initialize and train the perceptron
def train_perceptron(X: np.ndarray, y: np.ndarray) -> tuple:
    perceptron = Perceptron(
        input_size=2,
        activation_func=sigmoid,
        activation_deriv=sigmoid_derivative,
        error_func=mean_squared_error,
        learning_rate=0.1,
        l1_ratio=0,
        l2_ratio=0,
    )

    n_epochs = 100
    errors = []
    weights_over_epochs = []

    for epoch in range(n_epochs):
        perceptron.train(X, y, epochs=1, batch_size=1, verbose=False)
        output = perceptron.forward(X)
        error = mean_squared_error(y, output)
        errors.append(error)
        weights_over_epochs.append(np.copy(perceptron.weights))

    return perceptron, errors, weights_over_epochs


# Function to plot training results
def plot_results(
    errors: list,
    weights_over_epochs: list,
    perceptron: Perceptron,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    plt.figure(figsize=(12, 5))

    # Plotting the error over epochs
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.title("Error over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)

    # Plotting the decision boundary evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs_to_show = [0, int(len(errors) / 4), int(len(errors) / 2), len(errors) - 1]
    for i, epoch in enumerate(epochs_to_show):
        ax = axes[i // 2, i % 2]
        plot_decision_boundary(perceptron, X, y, epoch, weights_over_epochs[epoch], ax)

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

    perceptron, errors, weights_over_epochs = train_perceptron(X, y)
    plot_results(errors, weights_over_epochs, perceptron, X, y)
