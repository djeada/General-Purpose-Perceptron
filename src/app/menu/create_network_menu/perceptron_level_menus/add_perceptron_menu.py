from app.menu.menu_interface import AbstractMenu


class AddPerceptronMenu(AbstractMenu):
    def display_menu(self):
        print("\nConfiguring a new Perceptron")
        perceptron_config = {}

        try:
            perceptron_config["input_size"] = int(
                input("Enter input size for Perceptron: ")
            )
            perceptron_config["output_size"] = int(
                input("Enter output size for Perceptron: ")
            )
            perceptron_config["activation_func"] = self.get_activation_function_choice()
            perceptron_config["loss_func"] = self.get_loss_function_choice()
            perceptron_config["learning_rate"] = float(
                input("Enter learning rate for Perceptron (default 0.1): ") or 0.1
            )
            self.deactivate()
            return perceptron_config

        except ValueError:
            print("Invalid input. Please enter a valid number.")
            self.deactivate()
            return None

    def get_activation_function_choice(self):
        options = {
            "1": "sigmoid",
            "2": "basicsigmoid",
            "3": "relu",
            "4": "step",
            "5": "tanh",
            "6": "linear",
        }
        print("Select an activation function:")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice: ")
        return options.get(choice, "linear").lower()

    def get_loss_function_choice(self):
        options = {"1": "difference", "2": "mse", "3": "crossentropy", "4": "huber"}
        print("Select a loss function:")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice: ")
        return options.get(choice, "mse").lower()

    def process_choice(self, choice):
        # Implement logic for this menu
        pass
