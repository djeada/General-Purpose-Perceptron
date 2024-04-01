import logging
from abc import ABC
from typing import Dict, Union, Callable


class AbstractMenu(ABC):
    def __init__(
        self,
        name: str,
        menu_options: Dict[str, Dict[str, Union[str, Callable]]],
        parent_menu: "AbstractMenu" = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.menu_options = menu_options
        self.parent_menu = parent_menu
        self.is_active = True
        self.result = None

    def display_menu(self):
        """Displays the menu options."""
        print(f"\n--- {self.name} ---")
        for i, (key, option) in enumerate(self.menu_options.items()):
            print(f"{i + 1}. {option['text']}")

    def process_choice(self, choice: str):
        """Processes the user's choice and executes the corresponding action."""
        if choice in self.menu_options:
            self.menu_options[choice]["action"]()
        else:
            self.display_invalid_choice_message(choice)

    def display_invalid_choice_message(self, choice: str):
        """Displays an error message for invalid choices and lists valid options."""
        print(f"\nInvalid choice '{choice}'. Please enter a valid option.")
        print("Valid options are:")
        for key in self.menu_options.keys():
            print(f" - {key}")

    def run(self):
        """Runs the menu, displaying options and handling user input."""
        while self.is_active:
            self.display_menu()
            if not self.is_active:
                break
            choice = input("Enter your choice: ").strip()
            self.process_choice(choice)

    def deactivate(self) -> None:
        """Deactivates the menu."""
        self.is_active = False
