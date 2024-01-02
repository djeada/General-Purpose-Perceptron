from abc import ABC, abstractmethod


class Menu(ABC):
    def __init__(self, parent_menu=None):
        self.parent_menu = parent_menu

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def handle_selection(self, choice):
        pass

    def run(self):
        while True:
            self.display()
            choice = input("Enter your choice: ")
            if choice.lower() == "back" and self.parent_menu:
                return
            self.handle_selection(choice)
