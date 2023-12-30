import pytest
import numpy as np
from neura_command.neuron.neuron_interface import NeuronInterface

from src.neura_command.layer.layer import Layer


class MockNeuron(NeuronInterface):
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Assuming each neuron returns the sum of the input features for simplicity
        return X.sum(axis=1)  # This ensures a 1D array output per neuron

    def train(
        self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, verbose: bool
    ):
        pass  # Mock training


def test_layer_initialization():
    neurons = [MockNeuron(), MockNeuron()]
    layer = Layer(neurons)
    assert len(layer.neurons) == 2


def test_layer_forward_pass():
    neurons = [MockNeuron(), MockNeuron()]
    layer = Layer(neurons)
    X = np.array([[1, 2], [3, 4]])
    output = layer.forward(X)

    # Expect the output shape to be (number_of_samples, number_of_neurons)
    expected_shape = (X.shape[0], len(neurons))
    assert output.shape == expected_shape

    # Adjusted expected output verification
    # Each neuron output should be the sum of the input row
    expected_output = np.array([X.sum(axis=1), X.sum(axis=1)]).T
    np.testing.assert_array_equal(output, expected_output)


def test_empty_layer():
    layer = Layer([])
    assert len(layer.neurons) == 0


def test_output_shape():
    neurons = [MockNeuron(), MockNeuron()]
    layer = Layer(neurons)
    X = np.array([[1, 2], [3, 4]])
    output = layer.forward(X)
    assert output.shape == (X.shape[0], len(neurons))


def test_different_input_sizes():
    neurons = [MockNeuron(), MockNeuron()]
    layer = Layer(neurons)
    for size in [1, 5, 10]:
        X = np.random.rand(size, 2)
        output = layer.forward(X)
        assert output.shape == (X.shape[0], len(neurons))


def test_error_handling_invalid_input():
    neurons = [MockNeuron(), MockNeuron()]
    layer = Layer(neurons)
    with pytest.raises(ValueError):
        layer.forward(None)  # Assuming forward method raises ValueError for None input


def test_consistency_of_forward_pass():
    neurons = [MockNeuron(), MockNeuron()]
    layer = Layer(neurons)
    X = np.array([[1, 2], [3, 4]])
    output1 = layer.forward(X)
    output2 = layer.forward(X)
    np.testing.assert_array_equal(output1, output2)
