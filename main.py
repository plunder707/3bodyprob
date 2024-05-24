import torch
from chunk import ThreeBodySystem, simulate_with_chaos, plot_positions, animate_positions

if __name__ == "__main__":
    masses = torch.tensor([5.972e24, 1.989e30, 7.348e22])  # Earth, Sun, Moon
    initial_positions = torch.tensor([
        [0, 0, 0],  # Sun
        [1.496e11, 0, 0],  # Earth
        [1.496e11 + 3.844e8, 0, 0]  # Moon
    ], dtype=torch.float32)
    initial_velocities = torch.tensor([
        [0, 0, 0],  # Sun
        [0, 29.78e3, 0],  # Earth
        [0, 29.78e3 + 1.022e3, 0]  # Moon
    ], dtype=torch.float32)
    
    system = ThreeBodySystem(masses, initial_positions, initial_velocities)
    total_time = 365 * 24 * 3600  # One year in seconds
    dt = 60  # One-minute time steps
    tolerance = 1e-5  # Error tolerance for adaptive time stepping

    # Example with chaotic mixing
    chaos_func = lambda x, a, b: x + a * torch.sin(b * x)  # Example custom chaos function
    chaos_args = [0.01, 10.0]  # Example chaos function arguments

    positions = simulate_with_chaos(system, total_time, dt, tolerance, chaos_func, chaos_args)

    # Plotting the positions in 2D
    plot_positions(positions)

    # Plotting the positions in 3D
    plot_positions(positions, three_d=True)

    # Animating the positions in 2D
    animate_positions(positions)

    # Animating the positions in 3D
    animate_positions(positions, three_d=True)
