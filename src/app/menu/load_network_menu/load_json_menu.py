import json
from pathlib import Path

from app.menu.menu_interface import AbstractMenu


class LoadJsonMenu(AbstractMenu):
    def __init__(self, parent_menu: AbstractMenu):
        super().__init__(
            parent_menu=parent_menu,
            name="Load Network from JSON File",
            menu_options={
                "1": {
                    "text": "Enter the path to the JSON file.",
                    "action": self.prompt_for_path,
                },
                "back": {
                    "text": "Enter 'back' to return to the main menu.",
                    "action": self.deactivate,
                },
            },
        )
        self.result = None

    def prompt_for_path(self):
        file_path = input("Enter the path: ").strip()
        self.process_path(file_path)
        self.deactivate()

    def process_path(self, file_path):

        path = Path(file_path)
        if not path.exists():
            print("File not found. Please enter a valid path.")
            return
        if path.suffix.lower() != ".json":
            print("Please provide a path to a '.json' file.")
            return

        self.load_json(path)

    def load_json(self, file_path: Path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                self.result = data
                print(f"Successfully loaded network from {file_path}")
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            self.logger.error(f"JSON decode error for file {file_path}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            self.logger.error(f"Error loading file {file_path}: {e}")
