import argparse
from abc import ABC, abstractmethod
from pathlib import Path


class BaseCommand(ABC):
    def __init__(self, args):
        self.args = args

    @staticmethod
    @abstractmethod
    def add_arguments(parser):
        pass

    @abstractmethod
    def execute(self):
        """Subclasses must implement this method."""
        pass

    @staticmethod
    def file_path(path):
        if not Path(path).exists():
            raise argparse.ArgumentTypeError(f"The file {path} does not exist.")
        return path
