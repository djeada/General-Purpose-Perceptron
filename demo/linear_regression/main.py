import numpy as np
import matplotlib.pyplot as plt
from neura_command.neuron.perceptron import Perceptron
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# Activation and Error Functions
def linear_activation(x: np.ndarray) -> np.ndarray:
    return x


def linear_activation_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def mean_squared_error_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def prepare_data(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.rand(n, 1)
    y = 2 * X + 1 + np.random.normal(-0.1, 0.1, (n, 1))
    return X, y


def train_perceptron(X_train_scaled: np.ndarray, y_train: np.ndarray) -> Perceptron:
    perceptron = Perceptron(
        input_size=1,
        activation_func=linear_activation,
        activation_deriv=linear_activation_derivative,
        error_func=mean_squared_error_func,
        learning_rate=0.01,
        l1_ratio=0.00001,
        l2_ratio=0.00001,
    )
    perceptron.train(X_train_scaled, y_train, epochs=1000, batch_size=1, verbose=True)
    return perceptron


def evaluate_and_print_metrics(y_test: np.ndarray, predictions: np.ndarray):
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²) Score: {r_squared}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")


def visualize(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    perceptron: Perceptron,
):
    y_real = 2 * X_train_scaled + 1

    plt.figure(figsize=(10, 6))

    # Plot training and test data
    plt.scatter(
        X_train_scaled,
        y_train,
        color="blue",
        alpha=0.6,
        label="Train Data",
        marker="o",
    )
    plt.scatter(
        X_test_scaled,
        y_test,
        color="green",
        alpha=0.6,
        label="Test Data",
        marker="x",
    )

    # Plot the real function and perceptron fit
    plt.plot(
        X_train_scaled,
        y_real,
        color="cyan",
        linestyle="--",
        linewidth=2,
        label="Real Function: y = 2x + 1",
    )
    plt.plot(
        X_train_scaled,
        perceptron.forward(X_train_scaled),
        color="red",
        linewidth=2,
        label=f"Perceptron Fit: y = {perceptron.weights[1]:.2f}x + {perceptron.weights[0]:.2f}",
    )

    # Add labels and legend
    plt.title("Perceptron Linear Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

    # Customize the plot appearance
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    plt.legend(loc="upper left")

    # Show the plot
    plt.tight_layout()
    plt.show()


def main():
    X, y = prepare_data()
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    perceptron = train_perceptron(X_train_scaled, y_train)
    predictions = perceptron.forward(X_test_scaled)

    print("Real Function: y = 2x + 1")
    print(
        f"Perceptron Fit: y = {perceptron.weights[1]:0.2}x + {perceptron.weights[0]:0.2}"
    )

    evaluate_and_print_metrics(y_test, predictions)
    visualize(X_train_scaled, y_train, X_test_scaled, y_test, perceptron)


if __name__ == "__main__":
    main()
