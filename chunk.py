import torch
import math
from typing import Callable, List, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ThreeBodySystem:
    def __init__(self, masses: torch.Tensor, initial_positions: torch.Tensor, initial_velocities: torch.Tensor):
        """
        Initializes the ThreeBodySystem with masses, initial positions, and velocities.
        """
        self.masses = masses
        self.positions = initial_positions
        self.velocities = initial_velocities
        self.G = 6.67430e-11  # Gravitational constant

    def compute_accelerations(self) -> torch.Tensor:
        """
        Computes gravitational accelerations for the three-body system.
        """
        accelerations = torch.zeros_like(self.positions)
        for i in range(3):
            for j in range(3):
                if i != j:
                    r_ij = self.positions[j] - self.positions[i]
                    r = torch.norm(r_ij)
                    accelerations[i] += self.G * self.masses[j] * r_ij / r**3
        return accelerations

def runge_kutta_step(system: ThreeBodySystem, dt: float) -> None:
    """
    Performs a single Runge-Kutta integration step.
    """
    k1_v = system.compute_accelerations() * dt
    k1_x = system.velocities * dt

    k2_v = (system.compute_accelerations() + k1_v / 2) * dt
    k2_x = (system.velocities + k1_x / 2) * dt

    k3_v = (system.compute_accelerations() + k2_v / 2) * dt
    k3_x = (system.velocities + k2_x / 2) * dt

    k4_v = (system.compute_accelerations() + k3_v) * dt
    k4_x = (system.velocities + k3_x) * dt

    system.velocities += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    system.positions += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6

def adaptive_time_step(system: ThreeBodySystem, dt: float, tolerance: float) -> float:
    """
    Adjusts the time step size dynamically based on an error estimate.
    """
    original_positions = system.positions.clone()
    original_velocities = system.velocities.clone()

    runge_kutta_step(system, 2 * dt)
    positions_trial, velocities_trial = system.positions.clone(), system.velocities.clone()

    system.positions, system.velocities = original_positions, original_velocities
    runge_kutta_step(system, dt)
    runge_kutta_step(system, dt)

    error_estimate = torch.max(torch.abs(positions_trial - system.positions))

    if error_estimate > tolerance:
        dt *= 0.8
    elif dt > 1e3:
        dt = 1e3  # Prevent dt from becoming too large
    elif dt < 1e-6:
        dt = 1e-6  # Prevent dt from becoming too small
    else:
        dt *= 1.2

    return dt

def simulate(system: ThreeBodySystem, total_time: float, dt: float, tolerance: float) -> List[torch.Tensor]:
    """
    Simulates the three-body system over a specified total time.
    """
    num_steps = int(total_time / dt)
    positions = []
    for _ in range(num_steps):
        dt = adaptive_time_step(system, dt, tolerance)
        runge_kutta_step(system, dt)
        positions.append(system.positions.clone())
    return positions

def apply_chaotic_mixing(system: ThreeBodySystem, chaos_function: Callable, *chaos_args) -> None:
    """
    Applies chaotic mixing to the positions and velocities of the three-body system.
    """
    for i in range(3):
        system.positions[i] = chaos_function(system.positions[i], *chaos_args)
        system.velocities[i] = chaos_function(system.velocities[i], *chaos_args)

def simulate_with_chaos(system: ThreeBodySystem, total_time: float, dt: float, tolerance: float, 
                        chaos_function: Optional[Callable] = None, chaos_params: Optional[Union[List, torch.Tensor]] = None) -> List[torch.Tensor]:
    """
    Simulates the three-body system with optional chaotic mixing.
    """
    num_steps = int(total_time / dt)
    positions = []
    for _ in range(num_steps):
        dt = adaptive_time_step(system, dt, tolerance)
        runge_kutta_step(system, dt)
        if chaos_function is not None:
            apply_chaotic_mixing(system, chaos_function, *chaos_params)
        positions.append(system.positions.clone())
    return positions

def plot_positions(positions: List[torch.Tensor], three_d: bool = False) -> None:
    """
    Plots the 2D or 3D positions of the three-body system over time.
    """
    if three_d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(3):
            x_coords = [pos[i][0].item() for pos in positions]
            y_coords = [pos[i][1].item() for pos in positions]
            z_coords = [pos[i][2].item() for pos in positions]
            ax.plot(x_coords, y_coords, z_coords, label=f'Body {i+1}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
    else:
        plt.figure(figsize=(8, 8))
        for i in range(3):
            x_coords = [pos[i][0].item() for pos in positions]
            y_coords = [pos[i][1].item() for pos in positions]
            plt.plot(x_coords, y_coords, label=f'Body {i+1}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Three-Body Simulation with Chaotic Mixing')
        plt.legend()
        plt.grid(True)
    plt.show()

def animate_positions(positions: List[torch.Tensor], three_d: bool = False) -> None:
    """
    Creates an animation of the three-body simulation.

    Args:
        positions: A list of torch.Tensor objects containing the 3D positions 
                   of the three bodies at each time step.
        three_d (bool, optional): If True, creates a 3D animation. Defaults to False.
    """
    if three_d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes()
    
    lines = [ax.plot([], [], [], label=f'Body {i+1}')[0] if three_d else 
             ax.plot([], [], label=f'Body {i+1}')[0] for i in range(3)]

    def init():
        if three_d:
            # Set axis limits to encompass all positions during the simulation.
            ax.set_xlim3d(np.min(np.array(positions)[:, :, 0]), np.max(np.array(positions)[:, :, 0]))
            ax.set_ylim3d(np.min(np.array(positions)[:, :, 1]), np.max(np.array(positions)[:, :, 1]))
            ax.set_zlim3d(np.min(np.array(positions)[:, :, 2]), np.max(np.array(positions)[:, :, 2]))
        else:
            # Set axis limits to encompass all positions during the simulation.
            ax.set_xlim(np.min(np.array(positions)[:, :, 0]), np.max(np.array(positions)[:, :, 0]))
            ax.set_ylim(np.min(np.array(positions)[:, :, 1]), np.max(np.array(positions)[:, :, 1]))
        for line in lines:
            line.set_data([], [])
            if three_d:
                line.set_3d_properties([])
        return lines

    def update(frame):
        for i, line in enumerate(lines):
            x = [pos[i][0].item() for pos in positions[:frame]]
            y = [pos[i][1].item() for pos in positions[:frame]]
            line.set_data(x, y)
            if three_d:
                z = [pos[i][2].item() for pos in positions[:frame]]
                line.set_3d_properties(z)
        return lines

    ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, 
                        interval=200, repeat=False)  # Adjust interval for animation speed
    plt.legend()
    plt.grid(True)
    plt.show()
  
