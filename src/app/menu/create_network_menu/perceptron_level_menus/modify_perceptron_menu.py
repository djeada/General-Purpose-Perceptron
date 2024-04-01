from app.menu.menu_interface import AbstractMenu


class ModifyPerceptronMenu(AbstractMenu):
    def __init__(
        self,
        parent_menu: AbstractMenu,
        network_builder,
        layer_index: int,
        perceptron_index: int,
    ):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder
        self.layer_index = layer_index
        self.perceptron_index = perceptron_index

    def display_menu(self) -> None:
        """Displays the menu for modifying an existing Perceptron."""
        print("\n--- Modifying an existing Perceptron ---")
        print("1. Modify input size")
        print("2. Modify output size")
        print("3. Change activation function")
        print("4. Change loss function")
        print("5. Adjust learning rate")
        print("Enter 'back' to return to the previous menu.")

    def process_choice(self, choice: str) -> None:
        """Processes the user's choice for perceptron modification."""
        options = {
            "1": self.modify_input_size,
            "2": self.modify_output_size,
            "3": self.change_activation_function,
            "4": self.change_loss_function,
            "5": self.adjust_learning_rate,
            "back": self.deactivate,
        }

        action = options.get(choice)
        if action:
            action()
        else:
            print("Invalid choice. Please try again.")
            self.display_menu()

    def modify_input_size(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_input_size = int(
            input(f"Enter new input size (current: {perceptron['input_size']}): ")
        )
        perceptron["input_size"] = new_input_size
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Input size updated.")

    def modify_output_size(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_output_size = int(
            input(f"Enter new output size (current: {perceptron['output_size']}): ")
        )
        perceptron["output_size"] = new_output_size
        self.network_builder.set_perceptron(
            self.layer_index, self.perceptron_index, perceptron
        )
        print("Output size updated.")

    def change_activation_function(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_activation_func = self.get_activation_function_choice(
            current=perceptron["activation_func"]
        )
        perceptron["activation_func"] = new_activation_func
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Activation function updated.")

    def change_loss_function(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_loss_func = self.get_loss_function_choice(current=perceptron["loss_func"])
        perceptron["loss_func"] = new_loss_func
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Loss function updated.")

    def adjust_learning_rate(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_learning_rate = float(
            input(f"Enter new learning rate (current: {perceptron['learning_rate']}): ")
        )
        perceptron["learning_rate"] = new_learning_rate
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Learning rate updated.")

    def get_activation_function_choice(self, current: str):
        options = {
            "1": "sigmoid",
            "2": "basicsigmoid",
            "3": "relu",
            "4": "step",
            "5": "tanh",
            "6": "linear",
        }

        print(f"Select an activation function (current: {current}):")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice (leave blank for no change): ")
        if choice.strip() == "":
            return current
        return options.get(choice, current).lower()

    def get_loss_function_choice(self, current: str):
        options = {"1": "difference", "2": "mse", "3": "crossentropy", "4": "huber"}

        print(f"Select a loss function (current: {current}):")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice (leave blank for no change): ")
        if choice.strip() == "":
            return current
        return options.get(choice, current).lower()
