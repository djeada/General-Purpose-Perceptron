import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from neura_command.neuron.perceptron import Perceptron

# Generate a synthetic regression dataset.
def generate_dataset(
    n_samples: int = 100, n_features: int = 1, noise: float = 20, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    y = y.reshape(-1, 1)  # Reshape y to make it a 2D array
    return X, y


# Initialize the Perceptron model with specified parameters.
def initialize_perceptron(
    input_size: int, output_size: int, learning_rate: float
) -> Perceptron:
    return Perceptron(
        input_size=input_size,
        output_size=output_size,
        activation_func=lambda x: x,
        activation_deriv=lambda x: np.ones_like(x),
        error_func=lambda y_true, y_pred: y_true - y_pred,
        learning_rate=learning_rate,
        l1_ratio=0,
        l2_ratio=0,
    )


# Train the Perceptron model and return the error history.
def train_perceptron(
    perceptron: Perceptron, X_train: np.ndarray, y_train: np.ndarray, epochs: int
) -> list[float]:
    errors = []
    for epoch in range(epochs):
        output = perceptron.forward(X_train)
        error = perceptron.error_func(y_train, output)
        perceptron.backward(error)
        perceptron.update_weights()
        mse = np.mean(error ** 2)
        errors.append(mse)
    return errors


# Plot the training and testing data, regression fit, and error history.
def plot_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_range: np.ndarray,
    predictions: np.ndarray,
    errors: list[float],
) -> None:
    plt.figure(figsize=(12, 6))

    # Plot the training and testing data, and regression fit.
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.5)
    plt.scatter(X_test, y_test, color="green", label="Testing Data", alpha=0.5)
    plt.plot(X_range, predictions, color="red", label="Regression Fit")
    plt.xlabel("Input Feature")
    plt.ylabel("Target Value")
    plt.title("Perceptron Regression Fit")
    plt.legend()

    # Plot the error history.
    plt.subplot(1, 2, 2)
    plt.plot(errors, label="MSE over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Gradient Descent Error Reduction")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Demonstrate gradient descent regression.
def main() -> None:
    # Generate dataset and split it into training and testing sets.
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize Perceptron model and train it.
    perceptron = initialize_perceptron(input_size=1, output_size=1, learning_rate=0.01)
    errors = train_perceptron(perceptron, X_train, y_train, epochs=100)

    # Generate predictions and plot the results.
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    predictions = perceptron.forward(X_range)

    plot_results(X_train, y_train, X_test, y_test, X_range, predictions, errors)


if __name__ == "__main__":
    main()
