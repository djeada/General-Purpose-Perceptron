import argparse
from pathlib import Path

from app.commands.create_network_command import CreateNetworkCommand
from app.commands.predict_command import PredictCommand
from app.commands.train_network_command import TrainNetworkCommand


class CommandHandler:
    def __init__(self):
        self.parser = self._create_parser()
        self.args = self.parser.parse_args()
        self.command_map = {
            "create": CreateNetworkCommand,
            "train": TrainNetworkCommand,
            "predict": PredictCommand,
        }

    def _create_parser(self):
        parser = argparse.ArgumentParser(
            description="NeuraCommand: Neural Network Command Line Tool"
        )
        subparsers = parser.add_subparsers(title="commands", dest="command")
        subparsers.required = True

        # Create network command
        def file_path(path):
            if not Path(path).exists():
                raise argparse.ArgumentTypeError(f"The file {path} does not exist.")
            return path

        create_parser = subparsers.add_parser(
            "create", help="Create a new neural network from a configuration file"
        )
        create_parser.add_argument(
            "config",
            type=file_path,
            help="Path to the neural network configuration file",
        )
        create_parser.add_argument(
            "--output",
            type=str,
            default="network.pkl",
            help="Path to save the created network (default: network.pkl)",
        )

        train_parser = subparsers.add_parser(
            "train", help="Train a neural network from a specified file"
        )
        train_parser.add_argument(
            "network", type=file_path, help="Path to the neural network file"
        )
        train_parser.add_argument(
            "--input_x",
            type=file_path,
            required=True,
            help="Path to the input features file",
        )
        train_parser.add_argument(
            "--input_y",
            type=file_path,
            required=True,
            help="Path to the target values file",
        )
        train_parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Number of training epochs (default: 100)",
        )
        train_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite the existing network file after training",
        )

        predict_parser = subparsers.add_parser(
            "predict", help="Make predictions using a trained neural network"
        )
        predict_parser.add_argument(
            "network", type=file_path, help="Path to the neural network file"
        )
        predict_parser.add_argument(
            "--input",
            type=file_path,
            required=True,
            help="Path to the input file for making predictions",
        )
        return parser

    def execute_command(self):
        command = self.command_map.get(self.args.command)
        if command:
            command_instance = command(self.args)
            command_instance.execute()
