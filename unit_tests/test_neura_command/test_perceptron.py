import numpy as np
import pytest
from neura_command.neuron.perceptron import Perceptron


@pytest.fixture
def perceptron_instance():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def mean_squared_error(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    return Perceptron(
        input_size=2,
        output_size=1,
        activation_func=sigmoid,
        activation_deriv=sigmoid_derivative,
        error_func=mean_squared_error,
        learning_rate=0.01,
        l1_ratio=0.01,
        l2_ratio=0.01,
    )


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
    input_grad = perceptron_instance.backward(error)
    # The shape of input_grad will have one extra dimension due to the bias term
    assert input_grad.shape == (X_test.shape[0], X_test.shape[1] + 1)


def test_weight_update(perceptron_instance):
    initial_weights = np.copy(perceptron_instance.weights)
    perceptron_instance.last_gradient = np.ones_like(perceptron_instance.weights)
    perceptron_instance.update_weights()
    for i, weight in enumerate(perceptron_instance.weights):
        # Check if weights are updated (assuming learning rate is not zero)
        assert weight != initial_weights[i]


def test_training_process(perceptron_instance):
    X_train = np.array([[0.5, -0.2], [0.3, 0.8]])
    y_train = np.array([1, 0])
    initial_error = perceptron_instance.error_func(
        y_train, perceptron_instance.forward(X_train)
    )
    perceptron_instance.train(X_train, y_train, epochs=10, batch_size=1, verbose=False)
    final_error = perceptron_instance.error_func(
        y_train, perceptron_instance.forward(X_train)
    )
    assert final_error < initial_error  # Expect error to decrease after training


def test_prediction(perceptron_instance):
    X_test = np.array([[0.5, -0.2]])
    predictions = perceptron_instance.forward(X_test)
    assert predictions.shape == (1,)  # Ensure correct shape of predictions


def test_activation_function_application(perceptron_instance):
    X_test = np.array([[0.1, 0.2]])
    raw_output = np.dot(np.c_[np.ones((1, 1)), X_test], perceptron_instance.weights)
    expected_output = perceptron_instance.activation_func(raw_output)
    actual_output = perceptron_instance.forward(X_test)
    assert np.allclose(actual_output, expected_output)


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


def test_error_reduction_multiple_epochs(perceptron_instance):
    X_train = np.array([[0.5, 0.4], [0.6, 0.7]])
    y_train = np.array([1, 0])
    perceptron_instance.train(X_train, y_train, epochs=1, batch_size=1)
    error_after_first_epoch = perceptron_instance.error_func(
        y_train, perceptron_instance.forward(X_train)
    )
    perceptron_instance.train(X_train, y_train, epochs=10, batch_size=1)
    error_after_ten_epochs = perceptron_instance.error_func(
        y_train, perceptron_instance.forward(X_train)
    )
    assert error_after_ten_epochs < error_after_first_epoch
