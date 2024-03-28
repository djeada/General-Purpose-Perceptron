from app.menu.load_network_menu.load_json_menu import LoadJsonMenu
from app.menu.load_network_menu.load_pickle_menu import LoadPickleMenu
from app.menu.menu_interface import AbstractMenu


class LoadNetworkMenu(AbstractMenu):
    def display_menu(self):
        print("\n--- Load Network from Disk ---")
        print("1. Load network architecture from a JSON file")
        print("2. Load network object from a pickle file")
        print("Enter 'back' to return to the main menu.")
        # Other specific options related to loading a network can be added here

    def process_choice(self, choice):
        if choice == "1":
            self.load_from_json()
        elif choice == "2":
            self.load_from_pickle()
        elif choice.lower() == "back":
            self.parent_menu.run()
        else:
            print("Invalid choice. Please try again.")

    def load_from_json(self):
        load_json_menu = LoadJsonMenu(parent_menu=self)
        load_json_menu.run()
        self.parent_menu.run()

    def load_from_pickle(self):
        load_pickle_menu = LoadPickleMenu(parent_menu=self)
        load_pickle_menu.run()
        self.parent_menu.run()
