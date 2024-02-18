from app.menu.create_network_menu.layer_level_menus.modify_layer_menu import LayerMenu
from app.menu.menu_interface import Menu
from app.network_builder.network_builder import NetworkBuilder


class SelectLayerMenu(Menu):
    def __init__(self, parent_menu: Menu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder

    def display(self):
        print("\n--- Modifying a Layer in the Network ---")
        num_layers = self.get_num_layers()

        if num_layers == 0:
            print("No layers to modify.")
            self.go_back()
        else:
            self.prompt_layer_index(num_layers)

    def get_num_layers(self):
        return len(self.network_builder.config["layers"])

    def prompt_layer_index(self, num_layers):
        print(
            f"Enter the index of the layer to modify (0-based index).\n"
            f"Currently, there are {num_layers} layers in the network."
        )
        choice = input("Layer index: ")
        self.handle_selection(choice, num_layers)

    def handle_selection(self, choice, num_layers):
        if choice.lower() == "back":
            self.go_back()
        else:
            self.process_layer_choice(choice, num_layers)

    def process_layer_choice(self, choice, num_layers):
        try:
            layer_index = int(choice)
            if 0 <= layer_index < num_layers:
                self.modify_layer(layer_index)
            else:
                print(
                    f"Invalid layer index. Please enter a number between 0 and {num_layers - 1}."
                )
                self.prompt_layer_index(num_layers)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            self.prompt_layer_index(num_layers)

    def modify_layer(self, layer_index):
        layer_menu = LayerMenu(
            parent_menu=self,
            network_builder=self.network_builder,
            layer_index=layer_index,
        )
        layer_menu.run()

    def go_back(self):
        self.parent_menu.run()
