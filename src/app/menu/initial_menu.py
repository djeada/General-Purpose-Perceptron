import sys

from app.menu.create_network_menu import CreateNetworkMenu
from app.menu.load_network_menu import LoadNetworkMenu
from app.menu.menu_interface import Menu


class InitialMenu(Menu):
    def display(self):
        print("\nNeural Network CLI Application")
        print("1. Load Network from Disk")
        print("2. Create Network from Scratch")
        print("3. Exit")

    def handle_selection(self, choice):
        if choice == "1":
            sub_menu = LoadNetworkMenu(parent_menu=self)
            sub_menu.run()
        elif choice == "2":
            sub_menu = CreateNetworkMenu(parent_menu=self)
            sub_menu.run()
        elif choice == "3":
            sys.exit(0)
        else:
            print("Invalid choice. Please enter a valid option.")
