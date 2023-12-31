import numpy as np
from matplotlib import pyplot as plt
from neura_command.neuron.perceptron import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Activation Function
def step_activation(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, 0)


# Dummy Derivative (not used but required by Perceptron class)
def dummy_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def prepare_data(
    n_repeats: int = 100, gate_type: str = "AND"
) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    if gate_type == "AND":
        y = np.array([0, 0, 0, 1])
    elif gate_type == "OR":
        y = np.array([0, 1, 1, 1])
    else:
        raise ValueError("Unsupported gate_type. Use 'AND' or 'OR'.")

    X = np.tile(X, (n_repeats, 1))
    y = np.tile(y, n_repeats)
    return X, y


def train_perceptron(X_train: np.ndarray, y_train: np.ndarray) -> Perceptron:
    perceptron = Perceptron(
        input_size=2,
        output_size=1,
        activation_func=step_activation,
        activation_deriv=dummy_derivative,
        error_func=lambda a, b: a - b,
        learning_rate=0.01,
        l1_ratio=0,
        l2_ratio=0,
    )
    perceptron.train(X_train, y_train, epochs=100, batch_size=1)
    return perceptron


def evaluate_and_print_metrics(y_test: np.ndarray, predictions: np.ndarray):
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")


def plot_decision_boundary(
    perceptron: Perceptron, X: np.ndarray, y: np.ndarray, gate_type: str
):
    # Setting min and max values and giving some padding
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predicting values across the grid
    Z = perceptron.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting the contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

    # Adding contour lines for decision boundaries
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors="k")

    # Plotting the original points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdBu, edgecolor="k")

    # Adding labels and title based on gate type
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"{gate_type} Gate Decision Boundary")

    # Adding a colorbar for reference
    plt.colorbar()

    plt.show()


def main():
    gate_types = ["AND", "OR"]

    for gate_type in gate_types:
        print(f"Training and evaluating the {gate_type} gate...")

        # Prepare data for the selected gate
        X, y = prepare_data(gate_type=gate_type)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train the perceptron
        print("Training the perceptron...")
        perceptron = train_perceptron(X_train, y_train)

        # Make predictions and evaluate
        print("Making predictions and evaluating...")
        predictions = perceptron.forward(X_test)
        evaluate_and_print_metrics(y_test, predictions)

        # Plot decision boundary
        print(f"Plotting the {gate_type} gate decision boundary...")
        plot_decision_boundary(perceptron, X, y, gate_type)

        print(f"Finished evaluating the {gate_type} gate.\n")


if __name__ == "__main__":
    main()
