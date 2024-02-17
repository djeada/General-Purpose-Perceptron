from app.menu.menu_interface import Menu
from app.network_builder.network_builder import NetworkBuilder


class ModifyLayerMenu(Menu):
    def __init__(self, parent_menu: Menu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder

    def display(self):
        print("\nModifying a layer in the network")

        try:
            num_layers = len(self.network_builder.config["layers"])
            if num_layers == 0:
                print("No layers to modify.")
                self.parent_menu.display()
                return

            layer_index = int(
                input("Enter the index of the layer to modify (0-based index): ")
            )

            if 0 <= layer_index < num_layers:
                # Provide options for modification here
                # e.g., change the number of perceptrons, modify perceptron configuration, etc.
                pass  # Replace with actual modification logic

            else:
                print(
                    f"Invalid layer index. Please enter a number between 0 and {num_layers - 1}."
                )

            self.parent_menu.display()

        except ValueError:
            print("Invalid input. Please enter a valid number.")

    def handle_selection(self, choice):
        if choice.lower() == "back":
            self.parent_menu.display()
        else:
            self.display()
