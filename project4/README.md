# Allegro Hand Grasp Synthesis Project

This project provides tools and environments for grasp synthesis and multi-fingered inverse kinematics using the Allegro Hand in simulation. It is designed for research and experimentation in robotic manipulation, specifically focusing on grasping objects with a multi-fingered robotic hand.

## Project Structure

- `AllegroHandEnv.py`: Environment setup for the Allegro Hand simulation.
- `grasp_synthesis.py`: Core algorithms for grasp synthesis.
- `multifingered_ik.py`: Multi-fingered inverse kinematics solver.
- `utils.py`: Utility functions used throughout the project.
- `main.ipynb`, `main_cube.ipynb`: Example Jupyter notebooks demonstrating usage and experiments.
- `mujoco_menagerie/`: Contains MuJoCo assets and models for simulation.
- `pictures/`: Contains plots and images related to experiments and results.

## Requirements

- Python 3.7+
- MuJoCo
- NumPy
- Other dependencies as required in your code (see imports in Python files)

## Getting Started

1. **Install dependencies**:
   - Make sure MuJoCo is installed and properly configured.
   - Install required Python packages:
     ```bash
     pip install numpy mujoco
     ```
2. **Run experiments**:
   - Open `main.ipynb` or `main_cube.ipynb` in Jupyter Notebook to explore example usage and run experiments.

## Usage

- Use `AllegroHandEnv.py` to set up the simulation environment for the Allegro Hand.
- Implement or modify grasp synthesis algorithms in `grasp_synthesis.py`.
- Use `multifingered_ik.py` for solving inverse kinematics for multi-fingered hands.
- Refer to the notebooks for sample workflows and visualization of results.

## Results

Plots and images generated from experiments are stored in the `pictures/` directory.

## License

This project is for academic and research purposes. Please cite appropriately if used in publications.

## Acknowledgements

- EECS 206B course materials
- MuJoCo simulation environment
