import logging
from abc import ABC
from typing import Dict, Union, Callable


import logging
from abc import ABC, abstractmethod
from typing import Dict, Union, Callable, List


class AbstractMenu(ABC):
    def __init__(
        self,
        name: str,
        menu_options: Dict[str, Dict[str, Union[str, Callable]]] = None,
        parent_menu: "AbstractMenu" = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.menu_options = menu_options or {}
        self.parent_menu = parent_menu
        self.is_active = True
        self.result = None

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

    @abstractmethod
    def display_menu(self):
        pass

    @abstractmethod
    def process_choice(self, choice: str):
        pass


class ChoiceBasedMenu(AbstractMenu):
    def display_menu(self):
        """Displays the menu options for a choice-based menu."""
        print(f"\n--- {self.name} ---")
        for i, (key, option) in enumerate(self.menu_options.items()):
            print(f"{i + 1}. {option['text']}")

    def process_choice(self, choice: str):
        """Processes the user's choice for a choice-based menu."""
        if choice in self.menu_options:
            self.menu_options[choice]["action"]()
        else:
            self.display_invalid_choice_message(choice)

    def display_invalid_choice_message(self, choice: str):
        """Displays a message indicating the choice is invalid and lists available choices."""
        print(f"Invalid choice: '{choice}'. The available choices are:")
        for key, option in self.menu_options.items():
            print(f"  {key}. {option['text']}")


class StepBasedMenu(AbstractMenu):
    def __init__(self, *args, steps: List[Callable], **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = steps
        self.current_step_index = 0

    def display_menu(self):
        """Displays the current step for a step-based menu."""
        print(f"\n--- {self.name} ---")
        print(f"Step {self.current_step_index + 1}/{len(self.steps)}")

    def process_choice(self, choice: str):
        """Executes the current step in a step-based menu."""
        try:
            step_result = self.steps[self.current_step_index]()
            self.current_step_index += 1
            if self.current_step_index >= len(self.steps):
                self.deactivate()
        except Exception as e:
            print(f"Error during step execution: {e}")
            self.deactivate()

    def run(self):
        """Runs the step-based menu."""
        while self.is_active:
            self.display_menu()
            self.process_choice(None)
