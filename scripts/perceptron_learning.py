import numpy as np
import pandas as pd
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
    for i, (input_row, output_val) in enumerate(list(zip(X, outputs))):
        training_data.append(
            {
                "Iteration": iteration + 1,
                "Input": np.round(input_row, 4),
                "Weight": np.round(weights, 4),
                "Output": np.round(output_val, 4),
                "Error": np.round(error, 4)[i],
            }
        )


def main():
    X, y = create_dataset()
    perceptron = initialize_perceptron(input_size=1, learning_rate=0.1)
    training_df = train_perceptron(perceptron, X, y, n_iterations=10)
    print(training_df.to_string(index=False))


if __name__ == "__main__":
    main()
