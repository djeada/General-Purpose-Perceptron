from abc import ABC, abstractmethod


class AbstractMenu(ABC):
    def __init__(self, parent_menu: "AbstractMenu" = None):
        self.parent_menu = parent_menu
        self.is_active = True

    @abstractmethod
    def display_menu(self) -> None:
        """Abstract method to display the menu."""
        pass

    @abstractmethod
    def process_choice(self, choice: str) -> None:
        """Abstract method to handle the user's choice."""
        pass

    def run(self):
        """Runs the menu, displaying options and handling user input."""
        while self.is_active:
            self.display_menu()
            if not self.is_active:
                break
            choice = input("Enter your choice: ").strip()
            if choice.lower() == "back" and self.parent_menu:
                break
            self.process_choice(choice)

    def deactivate(self) -> None:
        """Method to deactivate the menu."""
        self.is_active = False
