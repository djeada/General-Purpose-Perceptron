from app.menu.create_network_menu.perceptron_level_menus.add_perceptron_menu import (
    AddPerceptronMenu,
)
from app.menu.menu_interface import AbstractMenu
from app.network_builder.network_builder import NetworkBuilder


class AddLayerMenu(AbstractMenu):
    def __init__(self, parent_menu: AbstractMenu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder

    def display_menu(self) -> None:
        """Displays the menu for adding a new layer to the network."""
        print("\nAdding a new layer to the network")

        try:
            num_perceptrons = self.get_num_perceptrons()
            same_config = self.get_same_config_answer()

            layer_config = self.create_layer_config(num_perceptrons, same_config)
            self.network_builder.add_layer(layer_config)
            print(f"Layer added successfully. {self.network_builder.get_layer(-1)}")
            self.deactivate()

        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def get_num_perceptrons(self) -> int:
        """Gets the number of perceptrons for the new layer."""
        return int(input("Enter the number of perceptrons in this layer: "))

    def get_same_config_answer(self) -> str:
        """Asks the user if the same configuration should be used for all perceptrons."""
        answer = (
            input("Use the same configuration for all perceptrons? (yes/no): ")
            .lower()
            .strip()
        )
        while answer not in ["yes", "y", "no", "n"]:
            answer = (
                input("Use the same configuration for all perceptrons? (yes/no): ")
                .lower()
                .strip()
            )
        return answer

    def create_layer_config(self, num_perceptrons: int, same_config: str) -> dict:
        """Creates the configuration for the new layer."""
        if same_config in ["yes", "y"]:
            perceptron_config = self.get_perceptron_config()
            if perceptron_config:
                return {"perceptrons": [perceptron_config] * num_perceptrons}
        else:
            return self.get_individual_perceptron_configs(num_perceptrons)
        return {"perceptrons": []}

    def get_perceptron_config(self):
        """Gets a single perceptron configuration."""
        add_perceptron_menu = AddPerceptronMenu()
        return add_perceptron_menu.display_menu()

    def get_individual_perceptron_configs(self, num_perceptrons: int) -> dict:
        """Gets configurations for each perceptron individually."""
        layer_config = {"perceptrons": []}
        for i in range(num_perceptrons):
            print(f"\nConfiguring Perceptron {i + 1}")
            perceptron_config = self.get_perceptron_config()
            if perceptron_config:
                layer_config["perceptrons"].append(perceptron_config)
        return layer_config

    def process_choice(self, choice: str) -> None:
        """Processes the user's choice."""
        if choice.lower() == "back":
            self.deactivate()
        else:
            self.display_menu()
