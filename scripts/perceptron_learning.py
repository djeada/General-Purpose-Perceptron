import numpy as np
import pandas as pd
from neura_command.perceptron import Perceptron
from typing import List, Tuple


def create_dataset() -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR problem
    return X, y


def initialize_perceptron(input_size: int, learning_rate: float) -> Perceptron:
    return Perceptron(
        input_size=input_size,
        activation_func=lambda x: 1 / (1 + np.exp(-x)),  # Sigmoid
        activation_deriv=lambda x: x * (1 - x),  # Sigmoid Derivative
        error_func=lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2),  # MSE
        learning_rate=learning_rate,
        l1_ratio=0,
        l2_ratio=0,
    )


def train_perceptron(
    perceptron: Perceptron, X: np.ndarray, y: np.ndarray, n_iterations: int
) -> pd.DataFrame:
    training_data = []
    for i in range(n_iterations):
        perceptron.train(X, y, epochs=1, batch_size=1, verbose=False)
        output = perceptron.forward(X)
        error = perceptron.error_func(y, output)
        record_training_progress(i, X, perceptron.weights, output, error, training_data)
    return pd.DataFrame(training_data)


def record_training_progress(
    iteration: int,
    X: np.ndarray,
    weights: np.ndarray,
    outputs: np.ndarray,
    error: float,
    training_data: List[dict],
):
    for input_row, output_val in zip(X, outputs):
        training_data.append(
            {
                "Iteration": iteration + 1,
                "Input": np.round(input_row, 2),
                "Weight": np.round(weights, 2),
                "Output": np.round(output_val, 2),
                "Error": round(error, 4),
            }
        )


def main():
    X, y = create_dataset()
    perceptron = initialize_perceptron(input_size=2, learning_rate=0.1)
    training_df = train_perceptron(perceptron, X, y, n_iterations=100)
    print(training_df.to_string(index=False))


if __name__ == "__main__":
    main()
