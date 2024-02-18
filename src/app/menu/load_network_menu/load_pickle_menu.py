from pathlib import Path

from app.menu.menu_interface import Menu


class LoadPickleMenu(Menu):
    def __init__(self, parent_menu):
        self.parent_menu = parent_menu

    def display(self):
        print("\nLoad Network from Pickle File")
        print("Enter the path to the pickle file:")
        print("Enter 'back' to return to the previous menu.")

    def load_json(self, file_path):
        print(f"Loading network from {file_path}...")
        # Add code to handle JSON file loading here

    def handle_selection(self, choice):
        file_path = Path(choice)

        if choice.lower() == "back":
            self.parent_menu.display()

        elif file_path.exists():
            self.load_json(file_path)
        else:
            print("File not found. Please enter a valid path.")
