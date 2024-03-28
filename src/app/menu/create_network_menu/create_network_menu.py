from app.menu.create_network_menu.layer_level_menus.add_layer_menu import AddLayerMenu
from app.menu.create_network_menu.layer_level_menus.select_layer_menu import (
    SelectLayerMenu,
)
from app.menu.create_network_menu.layer_level_menus.remove_layer_menu import (
    RemoveLayerMenu,
)
from app.menu.create_network_menu.save_network_menu import SaveNetworkMenu
from app.menu.menu_interface import AbstractMenu
from app.network_builder.network_builder import NetworkBuilder


class CreateNetworkMenu(AbstractMenu):
    def __init__(self, parent_menu: AbstractMenu):
        super().__init__(parent_menu=parent_menu)
        self.builder = NetworkBuilder()
        self.options = {
            "1": self.add_layer,
            "2": self.remove_layer,
            "3": self.modify_layer,
            "4": self.show_network,
            "5": self.save_network,
            "back": self.go_back,
        }

    def display_menu(self):
        print("\n--- Network Configuration Menu ---")
        print("1. Add Layer - Add a new layer to the neural network.")
        print("2. Remove Layer - Remove an existing layer from the network.")
        print(
            "3. Modify Existing Layer - Change the configuration of a specific layer."
        )
        print(
            "4. Show The Network - Display the current structure of the neural network."
        )
        print(
            "5. Save Network Configuration - Save the current network configuration to a file."
        )
        print("Enter 'back' to return to the main menu or 'exit' to quit.")

    def process_choice(self, choice):
        if choice.lower() in self.options:
            self.options[choice.lower()]()
        else:
            print("Invalid choice. Please try again.")
            self.run()

    def add_layer(self):
        add_layer_menu = AddLayerMenu(parent_menu=self, network_builder=self.builder)
        add_layer_menu.run()

    def remove_layer(self):
        remove_layer_menu = RemoveLayerMenu(
            parent_menu=self, network_builder=self.builder
        )
        remove_layer_menu.run()

    def modify_layer(self):
        modify_layer_menu = SelectLayerMenu(
            parent_menu=self, network_builder=self.builder
        )
        modify_layer_menu.run()

    def save_network(self):
        save_network_menu = SaveNetworkMenu(
            parent_menu=self, network_builder=self.builder
        )
        save_network_menu.run()

    def show_network(self):
        print(self.builder.to_json())

    def go_back(self):
        self.parent_menu.run()
