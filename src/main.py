from app.commands.command_handler import CommandHandler
from app.menu.intro_menu.initial_menu import InitialMenu
import sys


def main():
    if len(sys.argv) == 1:
        # No command-line arguments provided, default to interactive menu
        current_menu = InitialMenu()
        current_menu.run()
        return
    handler = CommandHandler()
    handler.execute_command()


if __name__ == "__main__":
    main()
