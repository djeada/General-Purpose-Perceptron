from app.menu.create_network_menu.perceptron_level_menus.add_perceptron_menu import (
    AddPerceptronMenu,
)
from app.menu.create_network_menu.perceptron_level_menus.modify_perceptron_menu import (
    ModifyPerceptronMenu,
)
from app.menu.menu_interface import Menu
from app.network_builder.network_builder import NetworkBuilder


class LayerMenu(Menu):
    def __init__(
        self, parent_menu: Menu, network_builder: NetworkBuilder, layer_index: int
    ):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder
        self.layer_index = layer_index
        self.options = {
            "1": self.add_perceptron,
            "2": self.remove_perceptron,
            "3": self.modify_perceptron,
            "4": self.display_layer,
            "back": self.go_back,
        }

    def display(self):
        print(f"\n--- Layer Configuration Menu for LAYER {self.layer_index} ---")
        print("1. Add Perceptron - Add a new perceptron to the layer.")
        print("2. Remove Perceptron - Remove an existing perceptron from the layer.")
        print(
            "3. Modify Perceptron - Change the configuration of a specific perceptron."
        )
        print("4. Display Layer - Show the current state of the layer.")
        print("Enter 'back' to return to the previous menu.")

    def handle_selection(self, choice):
        if choice.lower() in self.options:
            self.options[choice.lower()]()
        else:
            print("Invalid choice. Please try again.")
            self.display()

    def add_perceptron(self):

        add_perceptron_menu = AddPerceptronMenu()
        perceptron_config = add_perceptron_menu.display()
        if perceptron_config:
            layer = self.network_builder.get_layer(layer_index=self.layer_index)
            layer["perceptrons"].append(perceptron_config)
            self.network_builder.set_layer(
                layer_index=self.layer_index,
                new_layer_config=layer,
            )

    def remove_perceptron(self):
        layer = self.network_builder.get_layer(layer_index=self.layer_index)
        num_perceptrons = len(layer)

        if num_perceptrons == 0:
            print("No perceptrons to remove in this layer.")
            return

        try:
            index = int(
                input(
                    f"Enter index of perceptron to remove (0 to {num_perceptrons - 1}): "
                )
            )
            if not 0 <= index < num_perceptrons:
                print(
                    "Invalid index. Please enter a number between 0 and",
                    num_perceptrons - 1,
                )
                return
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return

        self.network_builder.remove_perceptron_from_layer(
            layer_index=self.layer_index, perceptron_index=index
        )
        print("Perceptron successfully removed.")
        self.display_layer()

    def modify_perceptron(self):
        menu = ModifyPerceptronMenu(parent_menu=self)
        menu.run()

    def display_layer(self):
        layer = self.network_builder.get_layer(layer_index=self.layer_index)
        print("Current layer configuration:")
        print(layer)

    def go_back(self):
        self.parent_menu.display()
