from app.menu.create_network_menu.add_perceptron_menu import AddPerceptronMenu
from app.menu.menu_interface import Menu
from app.network_builder.network_builder import NetworkBuilder


class AddLayerMenu(Menu):
    def __init__(self, parent_menu: Menu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder

    def display(self):
        print("\nAdding a new layer to the network")

        try:
            num_perceptrons = int(
                input("Enter the number of perceptrons in this layer: ")
            )
            same_config = (
                input("Use the same configuration for all perceptrons? (yes/no): ")
                .lower()
                .strip()
            )

            while same_config not in ["yes", "y", "no", "n"]:
                same_config = (
                    input("Use the same configuration for all perceptrons? (yes/no): ")
                    .lower()
                    .strip()
                )

            if same_config in ["yes", "y"]:
                add_perceptron_menu = AddPerceptronMenu()
                perceptron_config = add_perceptron_menu.display()
                if perceptron_config:
                    layer_config = {
                        "perceptrons": [perceptron_config] * num_perceptrons
                    }
            else:
                layer_config = {"perceptrons": []}
                for i in range(num_perceptrons):
                    print(f"\nConfiguring Perceptron {i + 1}")
                    add_perceptron_menu = AddPerceptronMenu()
                    perceptron_config = add_perceptron_menu.display()
                    if perceptron_config:
                        layer_config["perceptrons"].append(perceptron_config)

            self.network_builder.add_layer(layer_config)
            print(f"Layer added successfully. {self.network_builder.get_layer(-1)}")
            self.parent_menu.display()

        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def handle_selection(self, choice):
        if choice.lower() == "back":
            self.parent_menu.display()
        else:
            self.display()
