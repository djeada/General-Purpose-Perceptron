from app.menu.menu_interface import Menu
from app.network_builder.network_builder import NetworkBuilder


class RemoveLayerMenu(Menu):
    def __init__(self, parent_menu: Menu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder

    def display(self):
        print("\nRemoving a layer from the network")

        try:
            num_layers = len(self.network_builder.config["layers"])
            if num_layers == 0:
                print("No layers to remove.")
                self.parent_menu.display()
                return

            print(f"Current number of layers: {num_layers}")
            layer_index = int(
                input("Enter the index of the layer to remove (0-based index): ")
            )

            if 0 <= layer_index < num_layers:
                self.network_builder.remove_layer(layer_index)
                print(f"Layer at index {layer_index} removed successfully.")
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
