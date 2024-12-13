import logging
import numpy as np
import matplotlib.pyplot as plt
from galaxy import Galaxy
from perturber import Perturber
from simulation import Simulation
import os
import gc  # Import garbage collection module

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Change dpi setting for higher resolution plots
plt.rcParams['figure.dpi'] = 150


def main() -> None:
    # ============================================================
    # Initial Conditions (Dimensionless Units)
    # ============================================================

    # Number of stars
    N_stars = 100  # Increased number for better statistics

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
    initial_position_BH = np.array([5.0, 0.0, 4.0])  # Initial position [x, y, z]
    initial_velocity_BH = np.array([0.0, 0.05, -0.2])  # Initial velocity [vx, vy, vz]

    perturber1 = Perturber(mass=M_BH, position=initial_position_BH, velocity=initial_velocity_BH)

    # Set the perturber in the galaxy
    galaxy.set_perturbers(perturber1)

    # Compute an approximate orbital period at R=Rmax
    Omega_max = galaxy.omega(Rmax)
    T_orbit = 2 * np.pi / Omega_max  # Time for one orbit at Rmax

    # Total simulation time should be at least one orbital period at Rmax
    t_max = T_orbit * 1  # Simulate for 1 orbital period at Rmax

    # Time step array for test 
    dt_values = [1e1,1, 1e-1, 1e-2, 1e-3]  # Renamed variable to avoid overwriting

    # Select integrators to run: 'Leapfrog', 'RK4', or both
    selected_integrators = ['Leapfrog', 'RK4', 'Yoshida']  # Modify this list to select integrators

    # Initialize a dictionary to store energy differences for each integrator
    energy_diff_dict = {integrator: [] for integrator in selected_integrators}

    for dt in dt_values:

        logging.info(f"Running simulation with time step dt = {dt}")

        # Create Simulation instance with selected integrators
        simulation = Simulation(
            galaxy=galaxy,
            dt=dt,
            t_max=t_max,
            integrators=selected_integrators,
            paralellised=True
        )

        try:
            # Plot the galaxy potential before running the simulation
            simulation.plot_equipotential()

            # Run the simulation
            simulation.run()

            # Compute the absolute energy differences
            energy_differences = simulation.get_energy_difference()

            # Store the energy differences for each integrator
            for integrator, diff in energy_differences.items():
                if diff is not None:
                    energy_diff_dict[integrator].append((dt, diff))
                else:
                    energy_diff_dict[integrator].append((dt, np.nan))  # Use NaN for missing data

        except Exception as e:
            logging.error(f"An error occurred during simulation with dt={dt}: {e}")
            for integrator in selected_integrators:
                energy_diff_dict[integrator].append((dt, np.nan))
        finally:
            # ============================================================
            # Clear Cache After Each Simulation
            # ============================================================
            logging.info(f"Clearing cache after simulation with dt = {dt}")

            # Close all Matplotlib figures to free memory
            plt.close('all')

            # Delete simulation object and any arrays inside it
            del simulation

            # Explicitly clear NumPy arrays
            for name in dir():  # Loop through all variables
                obj = eval(name)
                if isinstance(obj, np.ndarray):  # Check if it's a NumPy array
                    del obj  # Delete the NumPy array

            # Perform garbage collection
            gc.collect()


    # ============================================================
    # Plot Energy Difference vs. Time Step (dt)
    # ============================================================

    logging.info("Generating Energy Difference vs. Time Step (dt) plot.")

    plt.figure(figsize=(10, 6))

    for integrator, data in energy_diff_dict.items():
        # Sort the data by dt in ascending order
        data_sorted = sorted(data, key=lambda x: x[0])
        dt_sorted, energy_diff_sorted = zip(*data_sorted)

        # Convert to numpy arrays for better handling
        dt_array = np.array(dt_sorted)
        energy_diff_array = np.array(energy_diff_sorted)

        # Plot on log-log scale
        plt.loglog(dt_array, energy_diff_array, marker='o', label=integrator, linewidth=2)

    plt.xlabel('Time Step (dt) (dimensionless)', fontsize=14)
    plt.ylabel('|E(t_f) - E(t_i)| (dimensionless)', fontsize=14)
    plt.title('Evolution of Absolute Energy Difference with Time Step for Each Integrator', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Save the plot
    # Ensure that the results directory exists
    results_dir = 'result'  # Define your results directory
    os.makedirs(results_dir, exist_ok=True)
    energy_diff_plot_path = os.path.join(results_dir, 'energy_difference_vs_dt.png')
    plt.savefig(energy_diff_plot_path)
    plt.close()
    logging.info(f"Energy Difference vs. Time Step plot saved to '{energy_diff_plot_path}'.")

    # Optional: Display the plot
    # plt.show()

    logging.info("Energy Difference vs. Time Step Analysis Completed.")






if __name__ == '__main__':
    main()
