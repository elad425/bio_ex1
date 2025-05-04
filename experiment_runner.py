import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Optional, Dict, Callable, Any

# Import the main BlockAutomaton class
from block_automaton import BlockAutomaton


def create_glider_pattern1(automaton):
    """Create first glider pattern for block automaton."""
    grid = np.zeros((automaton.size, automaton.size), dtype=int)
    center = automaton.size // 2

    pattern = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    x_start = center - 1
    y_start = center
    grid[x_start:x_start + pattern.shape[0], y_start:y_start + pattern.shape[1]] = pattern

    automaton.grid = grid
    return "Block-aligned glider pattern 1"


def create_glider_pattern2(automaton):
    """Create second glider pattern for block automaton."""
    grid = np.zeros((automaton.size, automaton.size), dtype=int)
    center = automaton.size // 2

    pattern2 = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])

    x_start = center - 1
    y_start = center - 1
    grid[x_start:x_start + pattern2.shape[0], y_start:y_start + pattern2.shape[1]] = pattern2

    automaton.grid = grid
    return "Block-aligned glider pattern 2"


def create_traffic_light_pattern(automaton):
    """Create patterns that oscillate like traffic lights for block automaton."""
    grid = np.zeros((automaton.size, automaton.size), dtype=int)
    center = automaton.size // 2

    checker1 = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0]
    ])

    x_start = center - (center % 2)
    y_start = center - (center % 2)
    grid[x_start:x_start + checker1.shape[0], y_start:y_start + checker1.shape[1]] = checker1

    automaton.grid = grid
    return "Block-aligned traffic light patterns"


def create_blinker_pattern(automaton):
    """Create patterns that oscillate between two states (blinkers)."""
    grid = np.zeros((automaton.size, automaton.size), dtype=int)
    center = automaton.size // 2

    pattern = np.array([
        [1, 1],
        [1, 1]
    ])

    x_start = center
    y_start = center
    p = pattern
    grid[x_start:x_start + p.shape[0], y_start:y_start + p.shape[1]] = p

    automaton.grid = grid
    return "Block-aligned blinker patterns"


def create_weird_pattern(automaton):
    """Create patterns that form complex structures."""
    grid = np.zeros((automaton.size, automaton.size), dtype=int)
    center = automaton.size // 2

    checker1 = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]
    ])

    x_start = center - (center % 2)
    y_start = center - (center % 2)
    grid[x_start:x_start + checker1.shape[0], y_start:y_start + checker1.shape[1]] = checker1

    automaton.grid = grid
    return "Complex square pattern"

def create_random_clusters_pattern(automaton):
    """Create random clusters of live cells."""
    # Start with a clean grid
    grid = np.zeros((automaton.size, automaton.size), dtype=int)

    # Create a few random clusters
    num_clusters = np.random.randint(3, 8)

    for _ in range(num_clusters):
        # Random cluster center
        cx = np.random.randint(10, automaton.size - 10)
        cy = np.random.randint(10, automaton.size - 10)

        # Random cluster size
        size = np.random.randint(3, 8)

        # Create cluster with higher probability in center
        for i in range(cx - size, cx + size):
            for j in range(cy - size, cy + size):
                if 0 <= i < automaton.size and 0 <= j < automaton.size:
                    # Higher probability near center, lower at edges
                    dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                    prob = 0.9 * np.exp(-0.3 * dist)
                    if np.random.random() < prob:
                        grid[i, j] = 1

    automaton.grid = grid
    return "Random clusters pattern"


class ExperimentalPatterns:
    """Class to run experiments with specific patterns for the block automaton."""

    def __init__(self):
        """Initialize the experimental patterns class."""
        self.patterns = {
            "glider1": create_glider_pattern1,
            "glider2": create_glider_pattern2,
            "traffic_light": create_traffic_light_pattern,
            "blinker": create_blinker_pattern,
            "weird": create_weird_pattern,
            "random_clusters": create_random_clusters_pattern
        }

    def run_experiment(
            self,
            pattern_name: str,
            size: int = 100,
            wrap: bool = True,
            generations: int = 100,
            animate: bool = True,
            show_grid: bool = False,
            save_output: bool = False
    ) -> Optional[BlockAutomaton]:
        """
        Run an experiment with a specific pattern.

        Args:
            pattern_name: Name of the pattern to use
            size: Size of the grid
            wrap: Whether to use wrap-around boundary conditions
            generations: Number of generations to run
            animate: Whether to show animation
            show_grid: Boolean for showing the grid or not
            save_output: Whether to save output files

        Returns:
            The automaton object or None if pattern not found
        """
        if pattern_name not in self.patterns:
            print(f"Unknown pattern: {pattern_name}")
            print(f"Available patterns: {', '.join(self.patterns.keys())}")
            return None

        # Create the automaton
        automaton = BlockAutomaton(size=size, wrap=wrap, initial_prob=0.5, show_grid=show_grid)

        # Apply the selected pattern
        pattern_description = self.patterns[pattern_name](automaton)
        print(f"Running experiment with pattern: {pattern_description}")
        print(f"Grid size: {size}x{size}, Wrap-around: {wrap}")

        if save_output:
            # Create output directory if it doesn't exist
            output_dir = f"output_{pattern_name}_{size}_wrap{wrap}"
            os.makedirs(output_dir, exist_ok=True)

            # Save initial state
            plt.figure(figsize=(10, 10))
            plt.imshow(automaton.grid, cmap=plt.cm.binary, interpolation='nearest')
            plt.title(f"Initial State: {pattern_description}")
            plt.savefig(f"{output_dir}/initial_state.png")
            plt.close()

        # Run the automaton
        if animate:
            automaton.run(generations=generations, animate=True)
        else:
            if save_output:
                # Save frames at intervals
                for _ in range(generations):
                    if _ % 5 == 0:  # Save every 5th generation
                        automaton.save_frame(f"{output_dir}/generation_{automaton.generation:04d}.png")
                    stability, population = automaton.step()
                    if _ % 10 == 0:
                        print(
                            f"Generation {automaton.generation}: Stability = {stability:.2f}%, Population = {population:.2f}%")

                # Plot and save metrics
                plt.figure(figsize=(12, 10))

                plt.subplot(2, 1, 1)
                plt.plot(range(len(automaton.stability_history)), automaton.stability_history)
                plt.title(f"Stability Over Generations - {pattern_description}")
                plt.xlabel("Generation")
                plt.ylabel("Stability (%)")
                plt.grid(True)

                plt.subplot(2, 1, 2)
                plt.plot(range(len(automaton.population_history)), automaton.population_history)
                plt.title("Population Over Generations")
                plt.xlabel("Generation")
                plt.ylabel("Population (%)")
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(f"{output_dir}/metrics.png")
                plt.close()

                # Save final state
                plt.figure(figsize=(10, 10))
                plt.imshow(automaton.grid, cmap=plt.cm.binary, interpolation='nearest')
                plt.title(f"Final State: {pattern_description}, Generation {automaton.generation}")
                plt.savefig(f"{output_dir}/final_state.png")
                plt.close()

                print(f"Experiment complete. Results saved to {output_dir}/")
            else:
                # Just run without saving
                for _ in range(generations):
                    stability, population = automaton.step()
                    if _ % 10 == 0:
                        print(
                            f"Generation {automaton.generation}: Stability = {stability:.2f}%, Population = {population:.2f}%")

                # Plot metrics
                automaton.plot_metrics()

        return automaton


def main():
    """Main function to run experiments based on command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Block Cellular Automaton Experiments')
    parser.add_argument('--pattern', type=str, required=True,
                        help='Pattern name to use')
    parser.add_argument('--size', type=int, default=100,
                        help='Size of the grid (NxN)')
    parser.add_argument('--wrap', action='store_true',
                        help='Use wrap-around boundary conditions')
    parser.add_argument('--gens', type=int, default=100,
                        help='Number of generations to run')
    parser.add_argument('--animate', action='store_true',
                        help='Show animation during execution')
    parser.add_argument('--list', action='store_true',
                        help='List available patterns')
    parser.add_argument('--show_grid', action='store_true',
                        help='Show grid')
    parser.add_argument('--save', action='store_true',
                        help='Save output files')

    args = parser.parse_args()

    experiments = ExperimentalPatterns()

    if args.list:
        print("Available patterns:")
        for pattern in experiments.patterns:
            print(f"  - {pattern}")
        return

    experiments.run_experiment(
        pattern_name=args.pattern,
        size=args.size,
        wrap=args.wrap,
        generations=args.gens,
        animate=args.animate,
        show_grid=args.show_grid,
        save_output=args.save
    )


if __name__ == "__main__":
    main()