import json

from neura_command.layer.layer import Layer
from neura_command.network.feed_forward_network.feed_forward_network import (
    FeedForwardNetwork,
)
from neura_command.network_utils.activation_functions import (
    LinearActivation,
    TanhActivation,
    StepActivation,
    ReLUActivation,
    BasicSigmoidActivation,
    SigmoidActivation,
)
from neura_command.network_utils.loss_functions import (
    HuberLoss,
    MSELoss,
    CrossEntropyLoss,
    DifferenceLoss,
)
from neura_command.neuron.perceptron.perceptron import Perceptron


class NetworkBuilder:
    def __init__(self, json_config=None):
        if isinstance(json_config, str):
            self.config = json.loads(json_config)
        elif isinstance(json_config, dict):
            self.config = json_config
        elif json_config is None:
            self.config = {"layers": []}
        else:
            raise ValueError("Invalid configuration format")

    def build(self):
        layers = []
        for layer_conf in self.config.get("layers", []):
            perceptrons = self._build_perceptrons(layer_conf.get("perceptrons", []))
            layers.append(Layer(perceptrons))
        return FeedForwardNetwork(layers)

    def _build_perceptrons(self, perceptrons_config):
        perceptrons = []
        for perceptron_conf in perceptrons_config:
            perceptron = Perceptron(
                input_size=perceptron_conf["input_size"],
                output_size=perceptron_conf["output_size"],
                activation_func=self._get_activation_function(
                    perceptron_conf["activation_func"]
                ),
                loss_func=self._get_loss_function(perceptron_conf["loss_func"]),
                learning_rate=perceptron_conf["learning_rate"],
            )
            perceptrons.append(perceptron)
        return perceptrons

    def _get_activation_function(self, name):
        # Return the activation function based on the name
        name = name.lower()
        if name == "sigmoid":
            return SigmoidActivation()
        elif name == "basicsigmoid":
            return BasicSigmoidActivation()
        elif name == "relu":
            return ReLUActivation()
        elif name == "step":
            return StepActivation()
        elif name == "tanh":
            return TanhActivation()
        elif name == "linear":
            return LinearActivation()
        else:
            print(
                f"Unknown activation function: {name}. Using LinearActivation as default."
            )
            return LinearActivation()

    def _get_loss_function(self, name):
        # Return the loss function based on the name
        name = name.lower()
        if name == "difference":
            return DifferenceLoss()
        elif name == "mse" or name == "meansquarederror":
            return MSELoss()
        elif name == "crossentropy":
            return CrossEntropyLoss()
        elif name == "huber":
            return HuberLoss()
        else:
            print(f"Unknown loss function: {name}. Using MSELoss as default.")
            return MSELoss()

    def add_layer(self, layer_config):
        self.config["layers"].append(layer_config)

    def get_layer(self, layer_index):
        if layer_index < len(self.config["layers"]):
            return self.config["layers"][layer_index]
        else:
            raise IndexError("Layer index out of range")

    def remove_layer(self, layer_index):
        if layer_index < len(self.config["layers"]):
            self.config["layers"].pop(layer_index)
        else:
            raise IndexError("Layer index out of range")

    def insert_layer(self, layer_index, layer_config):
        if layer_index <= len(self.config["layers"]):
            self.config["layers"].insert(layer_index, layer_config)
        else:
            raise IndexError("Layer index out of range")

    def add_perceptron_to_layer(self, layer_index, perceptron_config):
        if layer_index < len(self.config["layers"]):
            self.config["layers"][layer_index]["perceptrons"].append(perceptron_config)
        else:
            raise IndexError("Layer index out of range")

    def remove_perceptron_from_layer(self, layer_index, perceptron_index):
        if layer_index < len(self.config["layers"]):
            layer = self.config["layers"][layer_index]
            if perceptron_index < len(layer["perceptrons"]):
                layer["perceptrons"].pop(perceptron_index)
            else:
                raise IndexError("Perceptron index out of range")
        else:
            raise IndexError("Layer index out of range")

    def to_json(self):
        return json.dumps(self.config, indent=4)

    def from_json(self, json_config):
        self.config = json.loads(json_config)


if __name__ == "__main__":
    # Example usage
    builder = NetworkBuilder()
    builder.add_layer(
        {
            "perceptrons": [
                {
                    "input_size": 64,
                    "output_size": 10,
                    "activation_func": "1",
                    "loss_func": "2",
                    "learning_rate": 0.1,
                }
            ]
        }
    )
    builder.add_perceptron_to_layer(
        0,
        {
            "input_size": 10,
            "output_size": 20,
            "activation_func": "2",
            "loss_func": "1",
            "learning_rate": 0.1,
        },
    )
    print(builder.to_json())
