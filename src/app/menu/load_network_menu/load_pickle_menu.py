from pathlib import Path

from app.menu.menu_interface import AbstractMenu


class LoadPickleMenu(AbstractMenu):
    def __init__(self, parent_menu):
        self.parent_menu = parent_menu

    def display_menu(self):
        print("\nLoad Network from Pickle File")
        print("Enter the path to the pickle file:")
        print("Enter 'back' to return to the previous menu.")

    def load_json(self, file_path):
        print(f"Loading network from {file_path}...")
        # Add code to handle JSON file loading here

    def process_choice(self, choice):
        file_path = Path(choice)

        if choice.lower() == "back":
            self.parent_menu.display_menu()

        elif file_path.exists():
            self.load_json(file_path)
        else:
            print("File not found. Please enter a valid path.")
