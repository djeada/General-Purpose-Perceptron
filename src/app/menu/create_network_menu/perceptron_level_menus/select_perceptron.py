from app.menu.create_network_menu.perceptron_level_menus.modify_perceptron_menu import (
    ModifyPerceptronMenu,
)
from app.menu.menu_interface import AbstractMenu
from app.network_builder.network_builder import NetworkBuilder


class SelectPerceptronMenu(AbstractMenu):
    def __init__(
        self,
        parent_menu: AbstractMenu,
        network_builder: NetworkBuilder,
        layer_index: int,
    ):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder
        self.layer_index = layer_index

    def display_menu(self) -> None:
        """Displays the menu for selecting a perceptron to modify."""
        print(f"\n--- Modifying a Perceptron in Layer {self.layer_index} ---")
        num_perceptrons = self.get_num_perceptrons()

        if num_perceptrons == 0:
            print("No perceptrons to modify in this layer.")
            self.deactivate()
        else:
            self.prompt_perceptron_index(num_perceptrons)

    def get_num_perceptrons(self) -> int:
        """Returns the number of perceptrons in the selected layer."""
        layer = self.network_builder.get_layer(self.layer_index)
        return len(layer["perceptrons"])

    def prompt_perceptron_index(self, num_perceptrons: int) -> None:
        """Prompts the user to enter the index of the perceptron to modify."""
        print(
            f"Enter the index of the perceptron to modify (0-based index).\n"
            f"Currently, there are {num_perceptrons} perceptrons in this layer."
        )
        choice = input("Perceptron index: ")
        self.process_choice(choice, num_perceptrons)

    def process_choice(self, choice: str, num_perceptrons: int) -> None:
        """Processes the user's choice for a perceptron index."""
        if choice.lower() == "back":
            self.deactivate()
        else:
            self.process_perceptron_choice(choice, num_perceptrons)

    def process_perceptron_choice(self, choice: str, num_perceptrons: int) -> None:
        """Validates and processes the selected perceptron index."""
        try:
            perceptron_index = int(choice)
            if 0 <= perceptron_index < num_perceptrons:
                self.modify_perceptron(perceptron_index)
            else:
                print(
                    f"Invalid perceptron index. Please enter a number between 0 and {num_perceptrons - 1}."
                )
                self.prompt_perceptron_index(num_perceptrons)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            self.prompt_perceptron_index(num_perceptrons)

    def modify_perceptron(self, perceptron_index: int) -> None:
        """Initiates the modification of the selected perceptron."""
        modify_perceptron_menu = ModifyPerceptronMenu(
            parent_menu=self,
            network_builder=self.network_builder,
            layer_index=self.layer_index,
            perceptron_index=perceptron_index,
        )
        modify_perceptron_menu.run()
