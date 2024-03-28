from app.menu.menu_interface import AbstractMenu
from app.network_builder.network_builder import NetworkBuilder


class RemoveLayerMenu(AbstractMenu):
    def __init__(self, parent_menu: AbstractMenu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder

    def display_menu(self) -> None:
        """Displays the menu for removing a layer from the network."""
        print("\nRemoving a layer from the network")
        num_layers = len(self.network_builder.config["layers"])

        if num_layers == 0:
            print("No layers to remove.")
            self.deactivate()
        else:
            print(f"Current number of layers: {num_layers}")
            print(
                "Enter the index of the layer to remove (0-based index) or 'back' to go back."
            )

    def process_choice(self, choice: str) -> None:
        """Processes the user's choice for removing a layer or going back."""
        if choice.lower() == "back":
            self.deactivate()
        else:
            try:
                layer_index = int(choice)
                num_layers = len(self.network_builder.config["layers"])
                if 0 <= layer_index < num_layers:
                    self.network_builder.remove_layer(layer_index)
                    print(f"Layer at index {layer_index} removed successfully.")
                else:
                    print(
                        f"Invalid index. Please enter a number between 0 and {num_layers - 1}."
                    )
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            finally:
                self.deactivate()
