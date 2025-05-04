import os
import sys
import argparse
import time
from typing import Any

# Import the main modules
from block_automaton import BlockAutomaton
from experiment_runner import ExperimentalPatterns


class Menu:
    """
    Class to handle menu creation and navigation.
    """

    def __init__(self, title: str, options_generator=None, options=None, parent=None):
        self.title = title
        self.options_generator = options_generator
        self.options = options
        self.parent = parent

    def display(self) -> Any:
        """Display the menu and handle user selection."""
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')

            # Generate dynamic options if a generator is provided
            if self.options_generator:
                self.options = self.options_generator()

            # Print menu title and options
            print(f"\n===== {self.title} =====\n")

            for i, option in enumerate(self.options, 1):
                print(f"{i}. {option['text']}")

            # Add back option if this is a submenu
            if self.parent:
                print(f"\n{len(self.options) + 1}. Back")

            # Add quit option
            print(f"{len(self.options) + (2 if self.parent else 1)}. Quit")

            # Get user selection
            try:
                choice = int(input("\nEnter your choice: "))

                # Handle back option
                if self.parent and choice == len(self.options) + 1:
                    return "back"

                # Handle quit option
                if choice == len(self.options) + (2 if self.parent else 1):
                    print("\nExiting program. Goodbye!")
                    sys.exit(0)

                # Handle regular option
                if 1 <= choice <= len(self.options):
                    option = self.options[choice - 1]

                    # If the option has a submenu, display it
                    if "submenu" in option:
                        result = option["submenu"].display()
                        if result != "back":
                            return result
                    # If the option has an action, execute it
                    elif "action" in option:
                        return option["action"]()
                else:
                    print("\nInvalid choice. Please try again.")
                    time.sleep(1)
            except ValueError:
                print("\nPlease enter a number.")
                time.sleep(1)


class BlockCAMenu:
    """
    Main class for the Block Cellular Automaton menu system.
    """

    def __init__(self):
        """Initialize the menu system."""
        self.experiments = ExperimentalPatterns()
        self.settings = {
            "wrap": True,
            "size": 100,
            "generations": 100,
            "initial_prob": 0.5,
            "animate": True,
            "show_grid": False
        }

        # Get available patterns
        self.available_patterns = list(self.experiments.patterns.keys())

        # Create the main menu
        self.main_menu = self.create_main_menu()

    def create_main_menu(self) -> Menu:
        """Create the main menu structure."""
        # Create experiment type submenu
        experiment_menu = Menu("Select Experiment Type", options=[
            {"text": "Random Initialization (Part 1)", "action": self.run_random_experiment},
            {"text": "Predefined Patterns (Parts 2 & 3)", "submenu": self.create_patterns_menu()},
            {"text": "Settings", "submenu": self.create_settings_menu()}
        ], parent=None)

        # Create main menu
        main_menu = Menu("Block Cellular Automaton", options=[
            {"text": "Run Experiment", "submenu": experiment_menu},
            {"text": "Settings", "submenu": self.create_settings_menu()}
        ])

        return main_menu

    def create_patterns_menu(self) -> Menu:
        """Create the pattern's submenu."""
        pattern_options = []

        # We'll handle glider patterns specially to make them more user-friendly
        for pattern in self.available_patterns:
            if pattern == "glider1":
                display_name = "Glider Pattern 1"
            elif pattern == "glider2":
                display_name = "Glider Pattern 2"
            else:
                display_name = pattern.replace('_', ' ').title()

            pattern_options.append({
                "text": display_name,
                "action": lambda p=pattern: self.run_pattern_experiment(p)
            })

        return Menu("Select Pattern", options=pattern_options, parent=True)

    def generate_settings_options(self):
        """Generate dynamic settings menu options based on current settings."""
        return [
            {"text": f"Grid Size: {self.settings['size']}x{self.settings['size']}",
             "action": self.change_grid_size},
            {"text": f"Wrap-around: {'On' if self.settings['wrap'] else 'Off'}",
             "action": self.toggle_wrap},
            {"text": f"Generations: {self.settings['generations']}",
             "action": self.change_generations},
            {"text": f"Initial Probability: {self.settings['initial_prob']}",
             "action": self.change_initial_prob},
            {"text": f"Animation: {'On' if self.settings['animate'] else 'Off'}",
             "action": self.toggle_animation},
            {"text": f"Show Grid: {'On' if self.settings['show_grid'] else 'Off'}",
             "action": self.toggle_show_grid}
        ]

    def create_settings_menu(self) -> Menu:
        """Create the settings submenu."""
        return Menu("Settings", options_generator=self.generate_settings_options, parent=True)

    def run_random_experiment(self) -> str:
        """Run experiment with random initialization."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n===== Running Random Experiment =====\n")
        print(f"Grid Size: {self.settings['size']}x{self.settings['size']}")
        print(f"Wrap-around: {'On' if self.settings['wrap'] else 'Off'}")
        print(f"Initial Probability: {self.settings['initial_prob']}")
        print(f"Generations: {self.settings['generations']}")
        print(f"Animation: {'On' if self.settings['animate'] else 'Off'}")
        print(f"Show Grid: {'On' if self.settings['show_grid'] else 'Off'}")

        input("\nPress Enter to start...")

        # Create automaton
        automaton = BlockAutomaton(
            size=self.settings['size'],
            wrap=self.settings['wrap'],
            initial_prob=self.settings['initial_prob'],
            show_grid=self.settings['show_grid']
        )

        # Run automaton
        automaton.run(
            generations=self.settings['generations'],
            animate=self.settings['animate']
        )

        # If not animated, show metrics
        if not self.settings['animate']:
            automaton.plot_metrics()

        input("\nExperiment complete. Press Enter to continue...")
        return "back"

    def run_pattern_experiment(self, pattern: str) -> str:
        """Run experiment with a predefined pattern."""
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get a user-friendly name for the pattern
        if pattern == "glider1":
            display_name = "Glider Pattern 1"
        elif pattern == "glider2":
            display_name = "Glider Pattern 2"
        else:
            display_name = pattern.replace('_', ' ').title()

        print(f"\n===== Running {display_name} Experiment =====\n")
        print(f"Grid Size: {self.settings['size']}x{self.settings['size']}")
        print(f"Wrap-around: {'On' if self.settings['wrap'] else 'Off'}")
        print(f"Generations: {self.settings['generations']}")
        print(f"Animation: {'On' if self.settings['animate'] else 'Off'}")
        print(f"Show Grid: {'On' if self.settings['show_grid'] else 'Off'}")

        input("\nPress Enter to start...")

        # Run the pattern experiment
        self.experiments.run_experiment(
            pattern_name=pattern,
            size=self.settings['size'],
            wrap=self.settings['wrap'],
            generations=self.settings['generations'],
            animate=self.settings['animate'],
            show_grid=self.settings['show_grid']
        )

        input("\nExperiment complete. Press Enter to continue...")
        return "back"

    def change_grid_size(self) -> str:
        """Change the grid size setting."""
        try:
            size = int(input("\nEnter new grid size (even number): "))
            if size % 2 != 0:
                size += 1  # Make it even
            self.settings['size'] = max(10, min(200, size))  # Limit between 10 and 200
            print(f"\nGrid size set to {self.settings['size']}x{self.settings['size']}")
        except ValueError:
            print("\nInvalid input. Grid size unchanged.")

        time.sleep(1)
        return "back"

    def toggle_wrap(self) -> str:
        """Toggle wrap-around setting."""
        self.settings['wrap'] = not self.settings['wrap']
        print(f"\nWrap-around set to {'On' if self.settings['wrap'] else 'Off'}")
        time.sleep(1)
        return "back"

    def change_generations(self) -> str:
        """Change the number of generations."""
        try:
            gens = int(input("\nEnter number of generations: "))
            self.settings['generations'] = max(10, min(1000, gens))  # Limit between 10 and 1000
            print(f"\nGenerations set to {self.settings['generations']}")
        except ValueError:
            print("\nInvalid input. Generations unchanged.")

        time.sleep(1)
        return "back"

    def change_initial_prob(self) -> str:
        """Change the initial probability setting."""
        try:
            prob = float(input("\nEnter initial probability (0.0-1.0): "))
            self.settings['initial_prob'] = max(0.0, min(1.0, prob))  # Limit between 0 and 1
            print(f"\nInitial probability set to {self.settings['initial_prob']}")
        except ValueError:
            print("\nInvalid input. Initial probability unchanged.")

        time.sleep(1)
        return "back"

    def toggle_animation(self) -> str:
        """Toggle animation setting."""
        self.settings['animate'] = not self.settings['animate']
        print(f"\nAnimation set to {'On' if self.settings['animate'] else 'Off'}")
        time.sleep(1)
        return "back"

    def toggle_show_grid(self) -> str:
        """Toggle show grid setting."""
        self.settings['show_grid'] = not self.settings['show_grid']
        print(f"\nShow Grid set to {'On' if self.settings['show_grid'] else 'Off'}")
        time.sleep(1)
        return "back"

    def run(self):
        """Run the menu system."""
        self.main_menu.display()


def main():
    """Main function to start the menu system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Block Cellular Automaton Menu')
    parser.add_argument('--no-animation', action='store_true', help='Disable animation by default')
    args = parser.parse_args()

    # Create and run menu
    menu = BlockCAMenu()
    if args.no_animation:
        menu.settings['animate'] = False

    menu.run()


if __name__ == "__main__":
    main()