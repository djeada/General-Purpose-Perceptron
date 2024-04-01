from app.menu.menu_interface import AbstractMenu
from app.menu.load_network_menu.load_json_menu import LoadJsonMenu
from app.menu.load_network_menu.load_pickle_menu import LoadPickleMenu


class LoadNetworkMenu(AbstractMenu):
    def __init__(self, parent_menu: AbstractMenu):
        super().__init__(
            parent_menu=parent_menu,
            name="Load Network from Disk",
            menu_options={
                "1": {
                    "text": "Load network architecture from a JSON file",
                    "action": self.load_from_json,
                },
                "2": {
                    "text": "Load network object from a pickle file",
                    "action": self.load_from_pickle,
                },
                "back": {
                    "text": "Enter 'back' to return to the main menu.",
                    "action": self.deactivate,
                },
            },
        )

    def load_from_json(self):
        load_json_menu = LoadJsonMenu(parent_menu=self)
        load_json_menu.run()

    def load_from_pickle(self):
        load_pickle_menu = LoadPickleMenu(parent_menu=self)
        load_pickle_menu.run()
