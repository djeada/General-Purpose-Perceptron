import pytest
import numpy as np
from neura_command.perceptron import Perceptron


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
    X_bias_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    output = perceptron_instance._forward_pass(X_bias_test)
    assert output.shape == (1,)  # Ensure output shape is correct


def test_backward_pass(perceptron_instance):
    X_test = np.array([[0.5, -0.2]])
    y_test = np.array([1])
    X_bias_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    gradients = perceptron_instance._backward_pass(X_bias_test, y_test)
    assert (
        gradients.shape == perceptron_instance.weights.shape
    )  # Match shape of weights


def test_weight_update(perceptron_instance):
    initial_weights = np.copy(perceptron_instance.weights)
    gradients = np.ones_like(perceptron_instance.weights)
    perceptron_instance._update_weights(gradients)
    for i, weight in enumerate(perceptron_instance.weights):
        # Check if weights are updated (assuming learning rate is not zero)
        assert weight != initial_weights[i]


def test_training_process(perceptron_instance):
    X_train = np.array([[0.5, -0.2], [0.3, 0.8]])
    y_train = np.array([1, 0])
    initial_error = perceptron_instance.error_func(
        y_train,
        perceptron_instance._forward_pass(
            np.c_[np.ones((X_train.shape[0], 1)), X_train]
        ),
    )
    perceptron_instance.train(X_train, y_train, epochs=10, batch_size=1, verbose=False)
    final_error = perceptron_instance.error_func(
        y_train,
        perceptron_instance._forward_pass(
            np.c_[np.ones((X_train.shape[0], 1)), X_train]
        ),
    )
    assert final_error < initial_error  # Expect error to decrease after training


def test_prediction(perceptron_instance):
    X_test = np.array([[0.5, -0.2]])
    predictions = perceptron_instance.forward(X_test)
    assert predictions.shape == (1,)  # Ensure correct shape of predictions
