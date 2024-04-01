from app.menu.menu_interface import AbstractMenu


class UseNetworkMenu(AbstractMenu):
    def __init__(self, parent_menu: AbstractMenu):
        super().__init__(
            parent_menu=parent_menu,
            name="Use Network Menu",
            menu_options={
                "1": {"text": "Evaluate Data", "action": self.evaluate_data},
                "2": {"text": "Train Network", "action": self.train_network},
                "3": {
                    "text": "Show Network Statistics",
                    "action": self.show_statistics,
                },
                "back": {
                    "text": "Enter 'back' to return to the main menu.",
                    "action": self.deactivate,
                },
            },
        )

    def evaluate_data(self):
        # Add code for evaluating data
        print("Evaluating Data... [Placeholder]")

    def train_network(self):
        # Add code for training the network
        print("Training Network... [Placeholder]")

    def show_statistics(self):
        # Add code for showing network statistics
        print("Network Statistics... [Placeholder]")
