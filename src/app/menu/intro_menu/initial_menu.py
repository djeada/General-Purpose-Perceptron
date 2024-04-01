import sys

from app.menu.create_network_menu.create_network_menu import CreateNetworkMenu
from app.menu.load_network_menu.load_network_menu import LoadNetworkMenu
from app.menu.menu_interface import AbstractMenu
from app.menu.use_network_menu.use_network_menu import UseNetworkMenu


class InitialMenu(AbstractMenu):
    def __init__(self):
        super().__init__(
            name="Neura Command CLI Application",
            menu_options={
                "1": {"text": "Load Network from Disk", "action": self.load_network},
                "2": {
                    "text": "Create Network from Scratch",
                    "action": self.create_network,
                },
                "3": {"text": "Use Existing Network", "action": self.use_network},
                "4": {"text": "Exit", "action": self.exit_app},
            },
        )

    def load_network(self):
        sub_menu = LoadNetworkMenu(parent_menu=self)
        sub_menu.run()

    def create_network(self):
        sub_menu = CreateNetworkMenu(parent_menu=self)
        sub_menu.run()

    def use_network(self):
        sub_menu = UseNetworkMenu(parent_menu=self)  # Ensure you have this menu class
        sub_menu.run()

    def exit_app(self):
        sys.exit(0)
