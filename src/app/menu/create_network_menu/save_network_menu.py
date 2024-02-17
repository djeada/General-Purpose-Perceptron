import os

from app.menu.menu_interface import Menu
from app.network_builder.network_builder import NetworkBuilder


class SaveNetworkMenu(Menu):
    def __init__(self, parent_menu: Menu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder

    def display(self):
        print("\nSaving Network Configuration to Disk")

        file_path = input("Enter the file path to save the network configuration: ")

        # Additional checks can be added here (e.g., check if file already exists)

        try:
            with open(file_path, "w") as file:
                json_config = self.network_builder.to_json()
                file.write(json_config)
            print(f"Network configuration saved successfully to {file_path}")

        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

        self.parent_menu.display()
