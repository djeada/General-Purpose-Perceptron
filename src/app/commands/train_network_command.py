import logging

import numpy as np

from app.commands.base_command import BaseCommand
from neura_command.network.feed_forward_network.feed_forward_network_io import (
    FeedForwardNetworkIO,
)


class TrainNetworkCommand(BaseCommand):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model-pkl",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the neural network file (.pkl).",
        )
        parser.add_argument(
            "--features-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the CSV file containing the input features for training the model.",
        )
        parser.add_argument(
            "--targets-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the CSV file containing the target values corresponding to the input features.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Number of epochs to train the network. An epoch is one complete presentation of the data set to be learned.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Flag to indicate whether the trained model should overwrite the existing .pkl file.",
        )
        parser.add_argument(
            "--output-model-pkl",
            type=str,
            help="Custom path to save the trained network model file (.pkl). Meaningful if --overwrite is set to False.",
        )

    def execute(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Load the model
        try:
            model = FeedForwardNetworkIO.load_from_pickle(self.args.model_pkl)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return

        # Load training data
        try:
            features = np.loadtxt(self.args.features_csv, delimiter=",")
            targets = np.loadtxt(self.args.targets_csv, delimiter=",")
            logging.info("Training data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            return

        # Train the model
        try:
            model.train(features, targets, epochs=self.args.epochs)
            logging.info(f"Model trained successfully for {self.args.epochs} epochs.")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            return

        # Save the model
        output_path = (
            self.args.model_pkl if self.args.overwrite else self.args.output_model_pkl
        )
        if not output_path:
            logging.error("Output model path is not specified.")
            return
        try:
            FeedForwardNetworkIO.save_to_pickle(model, output_path)
            logging.info(f"Trained model saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving the trained model: {e}")
