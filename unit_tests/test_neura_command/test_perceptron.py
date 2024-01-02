import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from neura_command.network_utils.activation_functions import SigmoidActivation
from neura_command.network_utils.loss_functions import MSELoss
from neura_command.neuron.perceptron import Perceptron


@pytest.fixture
def perceptron_instance():

    return Perceptron(
        input_size=2,
        output_size=1,
        activation_func=SigmoidActivation(),
        loss_func=MSELoss(),
        learning_rate=0.01,
        l1_ratio=0.01,
        l2_ratio=0.01,
    )


@pytest.fixture
def regression_dataset():
    # Generate a synthetic regression dataset with 100 data points and 5 features
    X, y = make_regression(
        n_samples=100,
        n_features=2,
        noise=0.2,  # Add some noise to the target variable
        random_state=42,
    )
    return X, y


def test_initialization(perceptron_instance):
    assert perceptron_instance.learning_rate == 0.01
    assert perceptron_instance.l1_ratio == 0.01
    assert perceptron_instance.l2_ratio == 0.01
    assert len(perceptron_instance.weights) == 3  # 2 inputs + 1 bias


def test_forward_pass(perceptron_instance):
    X_test = np.array([[0.5, -0.2]])
    output = perceptron_instance.forward(X_test)
    assert output.shape == (1, 1)  # Ensure output shape is correct


def test_backward_pass(perceptron_instance):
    X_test = np.array([[0.5, -0.2]])
    y_test = np.array([1])
    perceptron_instance.forward(X_test)  # forward pass to set last_input
    error = y_test - perceptron_instance.forward(X_test)
    delta, input_grad = perceptron_instance.backward(error)
    # The shape of input_grad will have one extra dimension due to the bias term
    assert input_grad.T.shape == (X_test.shape[0], X_test.shape[1] + 1)


def test_weight_update(perceptron_instance):
    initial_weights = np.copy(perceptron_instance.weights)
    perceptron_instance.last_gradient = np.ones_like(perceptron_instance.weights)
    perceptron_instance.update_weights()
    for i, weight in enumerate(perceptron_instance.weights):
        # Check if weights are updated (assuming learning rate is not zero)
        assert weight != initial_weights[i]


def test_training_process(perceptron_instance, regression_dataset):
    X, y = regression_dataset

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    initial_error = perceptron_instance.error_func.compute(
        y_train, perceptron_instance.forward(X_train)
    )

    # Train the perceptron on the training data
    perceptron_instance.train(X_train, y_train, epochs=1, batch_size=1)

    final_error = perceptron_instance.error_func.compute(
        y_test, perceptron_instance.forward(X_test)
    )

    assert final_error < initial_error  # Expect error to decrease after training


def test_prediction(perceptron_instance):
    X_test = np.array([[0.5, -0.2]])
    predictions = perceptron_instance.forward(X_test)
    assert predictions.shape == (1, 1)  # Ensure correct shape of predictions


def test_gradient_calculation_in_backward(perceptron_instance):
    X_test = np.array([[0.5, 0.3]])
    y_test = np.array([0])
    perceptron_instance.forward(X_test)  # Forward pass
    error = y_test - perceptron_instance.forward(X_test)
    perceptron_instance.backward(error)
    # Check if gradient has been calculated and stored
    assert perceptron_instance.last_gradient is not None


def test_learning_rate_impact(perceptron_instance):
    perceptron_instance.learning_rate = 0.1
    initial_weights = np.copy(perceptron_instance.weights)
    perceptron_instance.last_gradient = np.ones_like(perceptron_instance.weights)
    perceptron_instance.update_weights()
    # Expect larger change in weights due to higher learning rate
    assert np.any(np.abs(perceptron_instance.weights - initial_weights) > 0.01)


def test_regularization_impact(perceptron_instance):
    initial_weights = np.copy(perceptron_instance.weights)
    perceptron_instance.l1_ratio = 0.1
    perceptron_instance.l2_ratio = 0.1
    perceptron_instance.last_gradient = np.zeros_like(perceptron_instance.weights)
    perceptron_instance.update_weights()
    # Expect weight change due to regularization even with zero gradient
    assert np.any(perceptron_instance.weights != initial_weights)


def test_error_reduction_multiple_epochs(perceptron_instance, regression_dataset):
    X, y = regression_dataset

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    initial_error = perceptron_instance.error_func.compute(
        y_test, perceptron_instance.forward(X_test)
    )

    errors = []  # List to store errors after each epoch

    # Train for multiple epochs
    for epoch in range(10):  # 10 epochs
        perceptron_instance.train(X_train, y_train, epochs=1, batch_size=1)
        error = perceptron_instance.error_func.compute(
            y_test, perceptron_instance.forward(X_test)
        )
        errors.append(error)

    # Check that error after 10 epochs is less than initial error
    assert errors[-1] < initial_error

    # Optionally, you can print or log the errors for analysis
    print("Errors after each epoch:", errors)
