import pytest
import numpy as np
from neura_command.layer.layer import Layer
from neura_command.network.network import Network
from neura_command.neuron.perceptron import Perceptron


def mock_loss_func(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mock_error_func(y_true, y_pred):
    return y_true - y_pred


def mock_activation_func(x):
    return x  # Simple identity function for activation


def mock_activation_deriv(x):
    return np.ones_like(x)  # Derivative of identity function


def mock_error_func(y_true, y_pred):
    return y_true - y_pred


def create_mock_perceptron():
    perceptron = Perceptron(
        input_size=3,
        output_size=2,
        activation_func=mock_activation_func,
        activation_deriv=mock_activation_deriv,
        error_func=mock_error_func,
        learning_rate=0.01,
        l1_ratio=0.01,
        l2_ratio=0.01,
    )
    perceptron.weights = np.ones(perceptron.input_size + 1)
    return perceptron


@pytest.fixture
def mock_perceptron():
    return create_mock_perceptron()


@pytest.fixture
def mock_network():
    layer = Layer([create_mock_perceptron() for i in range(3)])
    network = Network(loss_func=mock_error_func, error_func=mock_error_func)
    network.add_layer(layer)
    return network


def test_network_initialization():
    network = Network(loss_func=mock_loss_func, error_func=mock_error_func)
    assert network.layers == []


def test_network_add_layer(mock_network, mock_perceptron):
    new_layer = Layer([create_mock_perceptron() for i in range(3)])
    mock_network.add_layer(new_layer)
    assert len(mock_network.layers) == 2


def test_network_forward_pass(mock_network):
    X = np.array([[1, 2, 3]])
    output = mock_network.forward(X)
    assert output.size == len(mock_network.layers[-1])


def test_network_backward_pass(mock_network):
    error = np.array([[1, 2, 3]])
    mock_network.backward(error)


def test_network_update_weights(mock_network):
    learning_rate = 0.01
    mock_network.update_weights(learning_rate)
    # Check if weights in the layers have been updated appropriately.
    # This might involve examining the weights of each neuron.


def test_network_training(mock_network):
    X = np.array([[1, 2, 3]])
    Y = np.array([[2, 3, 4]])
    mock_network.train(X, Y, epochs=1, batch_size=1, learning_rate=0.01)
    # Check if the network has trained for one epoch.
    # You may want to check changes in weights or the loss value.
