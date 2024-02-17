from app.menu.intro_menu.initial_menu import InitialMenu
import argparse
import sys
from pathlib import Path


def create_network(args):
    # Load JSON configuration from args.config and create network
    # Save the created network to args.output
    pass


def train_network(args):
    # Load network from args.network
    # Train the network with data from args.input_x and args.input_y
    # If args.epochs is provided, use it; else, use default
    # If args.overwrite is True, overwrite the existing model after training
    pass


def predict(args):
    # Load network from args.network
    # Make predictions using data from args.input
    # Output predictions
    pass


def main():
    if len(sys.argv) == 1:
        # No command-line arguments provided, default to interactive menu
        current_menu = InitialMenu()
        current_menu.run()
        return

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
        "config", type=file_path, help="Path to the neural network configuration file"
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

    args = parser.parse_args()

    if args.command == "create":
        create_network(args)
    elif args.command == "train":
        train_network(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
