from app.menu.create_network_menu.perceptron_level_menus.add_perceptron_menu import (
    AddPerceptronMenu,
)
from app.menu.create_network_menu.perceptron_level_menus.select_perceptron import (
    SelectPerceptronMenu,
)
from app.menu.menu_interface import AbstractMenu
from app.network_builder.network_builder import NetworkBuilder


class LayerMenu(AbstractMenu):
    def __init__(
        self,
        parent_menu: AbstractMenu,
        network_builder: NetworkBuilder,
        layer_index: int,
    ):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder
        self.layer_index = layer_index
        self.options = {
            "1": self.add_perceptron,
            "2": self.remove_perceptron,
            "3": self.modify_perceptron,
            "4": self.display_layer,
            "back": self.deactivate,
        }

    def display_menu(self) -> None:
        """Displays the menu for layer configuration."""
        print(f"\n--- Layer Configuration Menu for LAYER {self.layer_index} ---")
        print("1. Add Perceptron - Add a new perceptron to the layer.")
        print("2. Remove Perceptron - Remove an existing perceptron from the layer.")
        print(
            "3. Modify Perceptron - Change the configuration of a specific perceptron."
        )
        print("4. Display Layer - Show the current state of the layer.")
        print("Enter 'back' to return to the previous menu.")

    def process_choice(self, choice: str) -> None:
        """Processes the user's choice."""
        action = self.options.get(choice.lower())
        if action:
            action()
        else:
            print("Invalid choice. Please try again.")
            self.display_menu()

    def add_perceptron(self) -> None:
        """Handles adding a new perceptron to the layer."""
        add_perceptron_menu = AddPerceptronMenu()
        perceptron_config = add_perceptron_menu.display_menu()
        if perceptron_config:
            layer = self.network_builder.get_layer(layer_index=self.layer_index)
            layer["perceptrons"].append(perceptron_config)
            self.network_builder.set_layer(
                layer_index=self.layer_index, new_layer_config=layer
            )

    def remove_perceptron(self) -> None:
        """Handles removing a perceptron from the layer."""
        layer = self.network_builder.get_layer(layer_index=self.layer_index)
        num_perceptrons = len(layer["perceptrons"])

        if num_perceptrons == 0:
            print("No perceptrons to remove in this layer.")
            return

        index = self.get_perceptron_index_to_remove(num_perceptrons)
        if index is not None:
            self.network_builder.remove_perceptron_from_layer(
                layer_index=self.layer_index, perceptron_index=index
            )
            print("Perceptron successfully removed.")
            self.display_layer()

    def modify_perceptron(self) -> None:
        """Initiates modification of a perceptron."""
        select_perceptron_menu = SelectPerceptronMenu(
            parent_menu=self,
            network_builder=self.network_builder,
            layer_index=self.layer_index,
        )
        select_perceptron_menu.run()

    def display_layer(self) -> None:
        """Displays the current configuration of the layer."""
        layer = self.network_builder.get_layer(layer_index=self.layer_index)
        print("Current layer configuration:")
        print(layer)

    def get_perceptron_index_to_remove(self, num_perceptrons: int) -> int:
        """Gets the index of the perceptron to be removed."""
        try:
            index = int(
                input(
                    f"Enter index of perceptron to remove (0 to {num_perceptrons - 1}): "
                )
            )
            if 0 <= index < num_perceptrons:
                return index
            print(
                f"Invalid index. Please enter a number between 0 and {num_perceptrons - 1}."
            )
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        return None
