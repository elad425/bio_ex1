import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def count_block_population(block):
    """Count the number of live cells in a 2x2 block."""
    return np.sum(block)


def flip_block(block):
    """Flip all values in a block (0->1, 1->0)."""
    return 1 - block


def rotate_block_180(block):
    """Rotate a 2x2 block by 180 degrees."""
    return np.rot90(block, 2)


def apply_block_rules(block):
    """Apply the rules to a single block."""
    live_cells = count_block_population(block)

    if live_cells == 2:
        # No change
        return block.copy()
    elif live_cells in [0, 1, 4]:
        # Flip all values
        return flip_block(block)
    elif live_cells == 3:
        # Flip all values and rotate 180°
        flipped = flip_block(block)
        return rotate_block_180(flipped)
    else:
        raise ValueError(f"Invalid number of live cells: {live_cells}")


class BlockAutomaton:
    """
    Block Cellular Automaton as described in the assignment.
    Alternates between two block structures and applies block-based rules.
    """

    def __init__(self, size=100, wrap=True, initial_prob=0.5, show_grid=False):
        """
        Initialize the automaton with given parameters.

        Args:
            size (int): Size of the grid (NxN)
            wrap (bool): Whether to use wrap-around boundary conditions
            initial_prob (float): Probability of a cell being alive in initial state
        """
        self.grid_lines = None
        self.animation = None
        self.show_grid = show_grid
        self.size = size
        self.wrap = wrap
        # Initialize grid randomly based on probability
        self.grid = np.random.random((size, size)) < initial_prob
        self.grid = self.grid.astype(int)

        # Metrics tracking
        self.stability_history = []
        self.population_history = []
        self.generation = 0

        # For visualization
        self.fig = None
        self.ax = None
        self.im = None

    def step(self):
        """
        Advance the automaton by one generation, applying rules to all blocks.
        Alternates between blue blocks (odd generations) and red blocks (even generations).
        """
        old_grid = self.grid.copy()
        new_grid = self.grid.copy()

        # Determine block structure based on generation parity
        is_odd_gen = self.generation % 2 == 0  # Start with blue blocks (gen 0 is considered even)

        # Process blocks
        for i in range(0, self.size, 2) if is_odd_gen else range(1, self.size, 2):
            for j in range(0, self.size, 2) if is_odd_gen else range(1, self.size, 2):
                if not self.wrap and (i + 1 >= self.size or j + 1 >= self.size):
                    continue  # Skip incomplete blocks at boundaries when not using wrap-around

                # Extract block with appropriate wrapping
                if self.wrap:
                    block = np.array([
                        [self.grid[i % self.size, j % self.size],
                         self.grid[i % self.size, (j + 1) % self.size]],
                        [self.grid[(i + 1) % self.size, j % self.size],
                         self.grid[(i + 1) % self.size, (j + 1) % self.size]]
                    ])
                else:
                    # Only process complete blocks
                    if i + 1 < self.size and j + 1 < self.size:
                        block = self.grid[i:i + 2, j:j + 2]
                    else:
                        continue

                # Apply rules to the block
                new_block = apply_block_rules(block)

                # Update grid with new block values
                if self.wrap:
                    new_grid[i % self.size, j % self.size] = new_block[0, 0]
                    new_grid[i % self.size, (j + 1) % self.size] = new_block[0, 1]
                    new_grid[(i + 1) % self.size, j % self.size] = new_block[1, 0]
                    new_grid[(i + 1) % self.size, (j + 1) % self.size] = new_block[1, 1]
                else:
                    new_grid[i:i + 2, j:j + 2] = new_block

        # Update grid and calculate metrics
        stability = 100 * (1 - np.sum(np.abs(new_grid - old_grid)) / (self.size * self.size))
        population = 100 * np.sum(new_grid) / (self.size * self.size)

        self.grid = new_grid
        self.generation += 1

        self.stability_history.append(stability)
        self.population_history.append(population)

        return stability, population

    def run(self, generations=250, animate=True, save_interval=None):
        """
        Run the automaton for a specified number of generations.

        Args:
            generations (int): Number of generations to run
            animate (bool): Whether to animate the evolution
            save_interval (int): If specified, save frames at this interval
        """
        if animate:
            self.setup_visualization()
            self.animation = FuncAnimation(
                self.fig, self.update, frames=generations, interval=10, blit=False
            )
            plt.show()
        else:
            # Run without animation
            for _ in range(generations):
                stability, population = self.step()

                # Print periodic updates
                if _ % 10 == 0:
                    print(f"Generation {self.generation}: Stability = {stability:.2f}%, Population = {population:.2f}%")

                # Save frames if requested
                if save_interval and _ % save_interval == 0:
                    self.save_frame(f"generation_{self.generation:04d}.png")

            # Plot metrics after run
            self.plot_metrics()

    def setup_visualization(self):
        """Set up the visualization plot."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.im = self.ax.imshow(
            self.grid,
            cmap=plt.cm.binary,
            interpolation='nearest',
            vmin=0,
            vmax=1
        )
        self.ax.set_title(f"Generation: {self.generation}")
        self.grid_lines = []
        if self.show_grid:
            self.draw_block_grid()  # Initial grid lines
        return self.im,

    def draw_block_grid(self):
        """Draw block grid lines depending on generation parity."""
        # Remove old lines
        for line in self.grid_lines:
            line.remove()
        self.grid_lines.clear()

        block_offset = 0 if self.generation % 2 == 0 else 1
        color = 'blue' if self.generation % 2 == 0 else 'red'
        linestyle = '-' if self.generation % 2 == 0 else '--'

        for i in range(block_offset, self.size, 2):
            line = self.ax.axhline(i - 0.5, color=color, linestyle=linestyle, linewidth=0.5)
            self.grid_lines.append(line)
            line = self.ax.axvline(i - 0.5, color=color, linestyle=linestyle, linewidth=0.5)
            self.grid_lines.append(line)

    def update(self, frame):
        """Update function for animation."""
        stability, population = self.step()

        self.im.set_array(self.grid)
        self.ax.set_title(f"Generation: {self.generation}, Stability: {stability:.2f}%, Population: {population:.2f}%")
        if self.show_grid:
            self.draw_block_grid()

        return self.im,

    def save_frame(self, filename):
        """Save the current state as an image."""
        if self.fig is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(
                self.grid,
                cmap=plt.cm.binary,
                interpolation='nearest',
                vmin=0,
                vmax=1
            )
            ax.set_title(f"Generation: {self.generation}")
            plt.savefig(filename)
            plt.close(fig)
        else:
            self.fig.savefig(filename)

    def plot_metrics(self):
        """Plot metrics over generations."""
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        # Plot stability history
        ax[0].plot(range(len(self.stability_history)), self.stability_history)
        ax[0].set_title("Stability Over Generations")
        ax[0].set_xlabel("Generation")
        ax[0].set_ylabel("Stability (%)")
        ax[0].grid(True)

        # Plot population history
        ax[1].plot(range(len(self.population_history)), self.population_history)
        ax[1].set_title("Population Over Generations")
        ax[1].set_xlabel("Generation")
        ax[1].set_ylabel("Population (%)")
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig("metrics.png")
        plt.show()


def main():
    """Main function to run the automaton based on command-line arguments."""
    parser = argparse.ArgumentParser(description='Run Block Cellular Automaton')
    parser.add_argument('--size', type=int, default=100, help='Size of the grid (NxN)')
    parser.add_argument('--wrap', action='store_true', help='Use wrap-around boundary conditions')
    parser.add_argument('--prob', type=float, default=0.5, help='Initial probability for live cells')
    parser.add_argument('--gens', type=int, default=250, help='Number of generations to run')
    parser.add_argument('--animate', action='store_true', help='Show animation')
    parser.add_argument('--show_grid', action='store_true', help='Show grid')

    args = parser.parse_args()

    # Create automaton
    automaton = BlockAutomaton(size=args.size, wrap=args.wrap, initial_prob=args.prob)

    # Run automaton
    automaton.run(generations=args.gens, animate=args.animate)


if __name__ == "__main__":
    main()
