import argparse

from app.commands.create_network_command import CreateNetworkCommand
from app.commands.predict_command import PredictCommand
from app.commands.train_network_command import TrainNetworkCommand


class CommandHandler:
    def __init__(self):
        self.command_map = {
            "create": CreateNetworkCommand,
            "train": TrainNetworkCommand,
            "predict": PredictCommand,
        }
        self.parser = self._create_parser()
        self.args, unknown_args = self.parser.parse_known_args()
        self._handle_arguments(unknown_args)

    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Neural Network Command Line Tool")
        parser.add_argument(
            "--action",
            type=str,
            choices=self.command_map.keys(),
            required=True,
            help="The action to perform (create, train, predict)",
        )
        return parser

    def _handle_arguments(self, unknown_args):
        command_class = self.command_map.get(self.args.action)
        if command_class:
            # Create a new subparser for the command
            subparser = argparse.ArgumentParser()
            command_class.add_arguments(subparser)
            self.command_args = subparser.parse_args(unknown_args)
        else:
            raise ValueError(f"Unknown command: {self.args.action}")

    def execute_command(self):
        command = self.command_map.get(self.args.action)
        if command:
            command_instance = command(self.command_args)
            command_instance.execute()
        else:
            print("No valid command provided.")
