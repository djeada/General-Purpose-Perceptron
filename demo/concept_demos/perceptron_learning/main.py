import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

from neura_command.neuron.perceptron import Perceptron


def create_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X = np.linspace(-1, 1, 10).reshape(-1, 1)  # Linearly spaced values
    y = 2 * X + np.random.normal(0, 0.2, X.shape)  # Linear relation with noise
    return X, y


def initialize_perceptron(input_size: int, learning_rate: float) -> Perceptron:
    return Perceptron(
        input_size=input_size,
        output_size=1,
        activation_func=lambda x: x,  # Linear activation
        activation_deriv=lambda x: 1,  # Derivative of linear activation
        error_func=lambda a, b: a - b,
        learning_rate=learning_rate,
        l1_ratio=0,
        l2_ratio=0,
    )


def train_perceptron(
    perceptron: Perceptron, X: np.ndarray, y: np.ndarray, n_iterations: int
) -> pd.DataFrame:
    training_data = []
    for i in range(n_iterations):
        perceptron.train(X, y, epochs=1, batch_size=1)
        output = perceptron.forward(X)
        error = perceptron.error_func(y, output)
        record_training_progress(
            i, X, y, perceptron.weights, output, error, training_data
        )
    return pd.DataFrame(training_data)


def record_training_progress(
    iteration: int,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    outputs: np.ndarray,
    error: float,
    training_data: List[dict],
):
    for i, (input_row, output_val, target_val) in enumerate(zip(X, outputs, y)):
        training_data.append(
            {
                "Iteration": iteration + 1,
                "Input": np.round(input_row, 4),
                "Weight": np.round(weights, 4),
                "Output": np.round(output_val, 4),
                "Target": np.round(target_val, 4),  # Added target column
                "Error": np.round(error, 4)[i],
            }
        )


def plot_fit(ax, X, y, weights, title):
    ax.scatter(X, y, color="blue", label="Data Points")
    line_x = np.linspace(X.min(), X.max(), 100)
    line_y = weights[0] * line_x + weights[1]
    ax.plot(line_x, line_y, color="red", label="Fitted Line")
    ax.set_title(title)
    ax.set_xlabel("Input X")
    ax.set_ylabel("Output y")
    ax.legend()


def main():
    X, y = create_dataset()
    perceptron = initialize_perceptron(input_size=1, learning_rate=0.1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot initial fit
    initial_weights = perceptron.weights.copy()
    plot_fit(axs[0], X, y, initial_weights, "Initial Fit")

    training_df = train_perceptron(perceptron, X, y, n_iterations=10)

    # Plot final fit
    final_weights = perceptron.weights.copy()
    plot_fit(axs[1], X, y, final_weights, "Final Fit after Training")

    plt.show()
    print(training_df.to_string(index=False))


if __name__ == "__main__":
    main()
