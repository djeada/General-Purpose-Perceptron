import sys

from app.menu.architecture_management_menu.architecture_managment_menu import (
    ArchitectureManagementMenu,
)
from app.menu.menu_interface import ChoiceBasedMenu
from app.menu.model_managment_menu.model_managment_menu import ModelManagmentMenu


class InitialMenu(ChoiceBasedMenu):
    def __init__(self):
        super().__init__(
            name="Neura Command CLI Application",
            menu_options={
                "1": {
                    "text": "Architecture Management",
                    "action": self.architecture_managment,
                },
                "2": {"text": "Model Management", "action": self.model_managment},
                "3": {"text": "Exit", "action": self.exit_app},
            },
        )

    def architecture_managment(self):
        sub_menu = ArchitectureManagementMenu(parent_menu=self)
        sub_menu.run()

    def model_managment(self):
        sub_menu = ModelManagmentMenu(parent_menu=self)
        sub_menu.run()

    def exit_app(self):
        sys.exit(0)
