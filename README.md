# Block Cellular Automaton – Assignment 1

## 🧠 Overview

This project implements a block-based 2D cellular automaton as described in the course "Computational Biology 80-512 / 89-512".

Each generation alternates between two types of 2x2 neighborhood blocks:
- Blue-lined blocks (even generations)
- Red dashed blocks (odd generations)

The automaton can be run with **wrap-around** or **non-wrap** boundary conditions.

## 📁 File Structure

| File | Description |
|------|-------------|
| `menu.py` | The main interactive menu system. Allows selecting experiments and changing settings. |
| `block_automaton.py` | Core automaton logic, rule application, animation, and metrics. |
| `experiment_runner.py` | Defines and runs experiments using random or predefined patterns. |
| `block_cellular_automaton_report.docx` | The full report with screenshots, analysis, and conclusions. |
| `menu.exe` | Standalone executable (if generated). No need to install Python. |

## 🚀 How to Run

### Using the Python Code
1. Install dependencies:
   ```
   pip install numpy matplotlib
   ```

2. Run the main menu:
   ```
   python menu.py
   ```

3. Use the interactive terminal menu to:
   - Run randomized experiments (Question 1)
   - Test gliders or predefined patterns (Questions 2 & 3)
   - Modify grid size, wrap-around, animation settings, etc.

### Using the Executable
If you prefer not to run Python:

1. Download `menu.exe` from here.
2. Double-click to launch the menu system.

No installation or internet required.

## ⚙️ Settings

You can toggle or edit the following options through the menu:
- Grid size (NxN, must be even)
- Wrap-around: On / Off
- Initial probability (for random live cells)
- Animation: On / Off
- Grid lines: Show / Hide
- Generations to simulate

## 📊 Metrics

During execution, the system records:
- **Stability**: % of unchanged cells from previous generation
- **Population**: % of live cells

These are plotted and/or saved after simulations.

## 📌 Notes

- Gliders, traffic light, blinker, and chaotic patterns are included.
- Wrap-around often allows richer, more sustained evolution.
- Stability and population metrics help assess convergence.

---

Created as part of the final submission for **Exercise 1**.
