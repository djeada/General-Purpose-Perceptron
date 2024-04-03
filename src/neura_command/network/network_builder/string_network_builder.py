from neura_command.layer.layer import Layer
from neura_command.network.feed_forward_network.feed_forward_network import (
    FeedForwardNetwork,
)
from neura_command.network_utils.activation_functions import *
from neura_command.network_utils.loss_functions import *
from neura_command.neuron.perceptron.perceptron import Perceptron


class StringNetworkBuilder:
    """A builder class for creating a feed-forward neural network."""

    def __init__(self):
        self.layers = []
        self.loss = MSELoss()

    def add_layer(self, input_size: int, output_size: int, activation: str):
        """Adds a layer to the network."""
        activation_func = self._get_activation_function(activation)
        perceptrons = [
            Perceptron(
                input_size,
                output_size,
                activation_func=activation_func,
                learning_rate=0.01,
            )
            for _ in range(output_size)
        ]
        self.layers.append(Layer(perceptrons))

    def _get_activation_function(self, name: str):
        """Retrieves an activation function based on its name."""
        functions = {
            "sigmoid": SigmoidActivation(),
            "basicsigmoid": BasicSigmoidActivation(),
            "relu": ReLUActivation(),
            "step": StepActivation(),
            "tanh": TanhActivation(),
            "linear": LinearActivation(),
        }
        if name not in functions:
            raise ValueError(f"Unknown activation function: {name}")
        return functions[name]

    def set_loss_function(self, name: str):
        """Sets the loss function for the network."""
        name = name.lower()
        functions = {
            "difference": DifferenceLoss(),
            "mse": MSELoss(),
            "meansquarederror": MSELoss(),
            "crossentropy": CrossEntropyLoss(),
            "huber": HuberLoss(),
        }
        self.loss = functions.get(name, MSELoss())

    def build(self):
        """Builds and returns the configured feed-forward network."""
        return FeedForwardNetwork(loss_func=self.loss, layers=self.layers)
