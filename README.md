# 3bodyprob
Three-Body Problem Simulation and Visualization in Python using PyTorch and Matplotlib

# Three-Body Problem Simulation

![Three-Body Simulation](https://github.com/plunder707/3bodyprob/images/three_body_simulation.png)

## Description

Welcome to the Three-Body Problem Simulation project! This repository contains a Python implementation of the three-body problem, a classic problem in celestial mechanics. The simulation models the gravitational interactions between three celestial bodies (e.g., the Sun, Earth, and Moon) and provides visualizations of their trajectories in both 2D and 3D.

The project uses PyTorch for numerical computations and Matplotlib for plotting and animation, making it an educational and visually appealing tool for exploring the dynamics of the three-body problem.

## Features

- **Numerical Integration**: Uses the Runge-Kutta method for accurate and stable numerical integration.
- **Adaptive Time Stepping**: Dynamically adjusts the time step size based on error estimates to maintain accuracy.
- **Chaotic Mixing**: Optionally applies chaotic functions to the system to illustrate complex and unpredictable behavior.
- **2D and 3D Visualization**: Generates static and animated plots of the bodies' trajectories in both 2D and 3D.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- `torch`
- `numpy`
- `matplotlib`

You can install these libraries using `pip`:

```bash
pip install torch numpy matplotlib

git clone https://github.com/plunder707/3bodyprob.git
cd 3bodyprob

python main.py
