# main.py

import logging
import numpy as np
import matplotlib.pyplot as plt
from galaxy import Galaxy
from perturber import Perturber
from simulation import Simulation

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Change dpi setting for higher resolution plots
plt.rcParams['figure.dpi'] = 150


def main() -> None:
    # ============================================================
    # Initial Conditions (Dimensionless Units)
    # ============================================================

    # Number of stars
    N_stars = 10000  # Increased number for better statistics

    # Maximum radial distance (Rmax) in dimensionless units
    Rmax = 10.0  # Adjust based on the simulation needs

    # Logging simulation properties
    logging.info("Starting the simulation with the following properties:")

    # Create Galaxy instance
    galaxy = Galaxy(mass=1.0, a=2.0, b=0.1, epsilon=0.1)

    # Initialize stars with the Schwarzschild velocity distribution
    galaxy.initialize_stars(N=N_stars, Rmax=Rmax, alpha=0.05, max_iterations=100, use_schwarzschild=True)

    # Create Perturber instance
    M_BH = 1*0.07  # Mass of the perturber (normalized)
    initial_position_BH = np.array([5.0, 0.0, 8.0])  # Initial position [x, y, z]
    initial_velocity_BH = np.array([0.0, 0.0, -0.4])  # Initial velocity [vx, vy, vz]

    perturber1 = Perturber(mass=M_BH, position=initial_position_BH, velocity=initial_velocity_BH)
    perturber2 = Perturber(mass=M_BH, position=-0.7*initial_position_BH, velocity=-0.5*initial_velocity_BH)

    # Set the perturber in the galaxy
    galaxy.set_perturbers(perturber1)

    # Compute an approximate orbital period at R=Rmax
    Omega_max = galaxy.omega(Rmax)
    T_orbit = 2 * np.pi / Omega_max  # Time for one orbit at Rmax

    # Total simulation time should be at least one orbital period at Rmax
    t_max = T_orbit * 1  # Simulate for 1 orbital period at Rmax

    # Time step
    dt = 0.1  # Smaller time step for better accuracy

    # Select integrators to run: 'Leapfrog', 'RK4', or both
    selected_integrators = ['Yoshida']  # Modify this list to select integrators

    # Create Simulation instance with selected integrators
    simulation = Simulation(galaxy=galaxy, dt=dt, t_max=t_max, integrators=selected_integrators, paralellised=True)

    # Plot the galaxy potential before running the simulation
    simulation.plot_equipotential()

    # Run the simulation
    simulation.run()

    # ============================================================
    # Generate Plots
    # ============================================================

    # Generate plots using the Simulation class methods
    simulation.plot_trajectories(subset=200)  # Plot a subset of 200 stars for clarity
    simulation.plot_galaxy_snapshots(n_snapshots=4, independantFig=False)  # Plot int(t_max/dt) snapshots independantly or not
    simulation.plot_rotation_curve()
    simulation.plot_energy_error()
    simulation.plot_angular_momentum_error()
    simulation.plot_execution_time()

    # Compute and compare velocity dispersions
    simulation.compute_velocity_dispersions()
    simulation.compute_velocity_dispersions_continuous()

    # Plot velocity histograms
    simulation.plot_velocity_histograms(subset=200)

    # Compute and log differences between integrators
    simulation.log_integrator_differences()

    # Save positions and velocities for each integrator
    """
    for integrator_name in selected_integrators:
        np.save(os.path.join(simulation.results_dir, f'positions_{integrator_name.lower()}.npy'),
                simulation.positions[integrator_name])
        np.save(os.path.join(simulation.results_dir, f'velocities_{integrator_name.lower()}.npy'),
                simulation.velocities[integrator_name])

    logging.info("All simulation data and plots have been saved.")
    """


if __name__ == '__main__':
    main()
