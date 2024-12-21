import logging
import numpy as np
import matplotlib.pyplot as plt
from galaxy import Galaxy
from perturber import Perturber
from simulation import Simulation
import os
import gc  # Import garbage collection module
import copy  # Import the copy module for deepcopy

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Matplotlib to use LaTeX for all text elements
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def main() -> None:
    # ============================================================
    # Simulation Parameters
    # ============================================================

    # Number of stars
    N_stars = 100  # Increased number for better statistics

    # Maximum radial distance (Rmax) in dimensionless units
    Rmax = 10.0  # Adjust based on the simulation needs

    # Mass of the perturber (normalized)
    M_BH = 1 * 0.07  

    # Initial position and velocity of the perturber
    initial_position_BH = np.array([5.0, 0.0, 4.0])  # [x, y, z]
    initial_velocity_BH = np.array([0.0, 0.05, -0.2])  # [vx, vy, vz]

    # Perturber instance (to be deep-copied if needed)
    base_perturber = Perturber(mass=M_BH, position=initial_position_BH, velocity=initial_velocity_BH)

    # Time step array for tests
    dt_values = [1, 1e-1, 1e-2, 1e-3]  # Renamed variable to avoid overwriting

    # Number of runs per dt to gather statistics
    n_runs = 3  # Adjust based on computational resources

    # Select integrators to run: 'Leapfrog', 'RK4', or both
    selected_integrators = ['Leapfrog', 'RK4', 'Yoshida']  # Modify this list to select integrators

    # Initialize a dictionary to store energy differences for each integrator and dt
    # Structure: energy_diff_dict[integrator][dt] = list of diffs
    energy_diff_dict = {integrator: {dt: [] for dt in dt_values} for integrator in selected_integrators}

    for dt in dt_values:
        logging.info(f"Running simulations with time step dt = {dt}")

        for run in range(1, n_runs + 1):
            logging.info(f"  Run {run}/{n_runs} for dt = {dt}")

            try:
                # ============================================================
                # Initialize Galaxy and Perturber for Each Run
                # ============================================================

                # Create a fresh Galaxy instance
                galaxy = Galaxy(mass=1.0, a=2.0, b=0.1, epsilon=0.1)

                # Initialize stars with the Schwarzschild velocity distribution
                galaxy.initialize_stars(N=N_stars, Rmax=Rmax, alpha=0.05, max_iterations=100, use_schwarzschild=False)

                # Create a fresh Perturber instance (deep copy if necessary)
                perturber = copy.deepcopy(base_perturber)

                # Set the perturber in the galaxy
                galaxy.set_perturbers(perturber)

                # Compute an approximate orbital period at R=Rmax for the current galaxy
                Omega_max = galaxy.omega(Rmax)
                T_orbit = 2 * np.pi / Omega_max  # Time for one orbit at Rmax

                # Total simulation time should be at least one orbital period at Rmax
                t_max = T_orbit * 1  # Simulate for 1 orbital period at Rmax

                # Create a fresh Simulation instance for each run to avoid state carry-over
                simulation = Simulation(
                    galaxy=copy.deepcopy(galaxy),  # Use deepcopy to create an independent copy
                    dt=dt,
                    t_max=t_max,
                    integrators=selected_integrators,
                    paralellised=True
                )

                # Plot the galaxy potential before running the simulation
                simulation.plot_equipotential()

                # Run the simulation
                simulation.run()

                # Compute the absolute energy differences
                energy_differences = simulation.get_energy_difference()

                # Store the energy differences for each integrator
                for integrator, diff in energy_differences.items():
                    if diff is not None:
                        energy_diff_dict[integrator][dt].append(diff)
                    else:
                        energy_diff_dict[integrator][dt].append(np.nan)  # Use NaN for missing data

            except Exception as e:
                logging.error(f"An error occurred during simulation run {run} with dt={dt}: {e}")
                for integrator in selected_integrators:
                    energy_diff_dict[integrator][dt].append(np.nan)
            finally:
                # ============================================================
                # Clear Cache After Each Simulation
                # ============================================================
                logging.info(f"  Clearing cache after simulation run {run} with dt = {dt}")

                # Close all Matplotlib figures to free memory
                plt.close('all')

                # Delete simulation object and any arrays inside it
                del simulation

                # Perform garbage collection
                gc.collect()

    # ============================================================
    # Compute Statistics for Energy Differences
    # ============================================================

    # Initialize dictionaries to store mean and std deviation
    energy_diff_mean = {integrator: [] for integrator in selected_integrators}
    energy_diff_std = {integrator: [] for integrator in selected_integrators}

    # Define a small tolerance to exclude near-zero values (optional)
    zero_tolerance = 1e-17

    for integrator in selected_integrators:
        for dt in dt_values:
            diffs = np.array(energy_diff_dict[integrator][dt])
            # Remove NaN values before computing statistics
            diffs = diffs[~np.isnan(diffs)]
            # Remove zero or near-zero values based on tolerance
            diffs = diffs[np.abs(diffs) > zero_tolerance]
            if len(diffs) > 0:
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
            else:
                mean_diff = np.nan
                std_diff = np.nan
            energy_diff_mean[integrator].append(mean_diff)
            energy_diff_std[integrator].append(std_diff)

    # ============================================================
    # Plot Energy Difference vs. Time Step (dt) with Error Bars
    # ============================================================

    logging.info("Generating Energy Difference vs. Time Step (dt) plot with error bars.")

    # Create figure and axes using subplots
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define machine epsilon for double precision
    machine_epsilon = np.finfo(float).eps

    for integrator in selected_integrators:
        mean_diffs = np.array(energy_diff_mean[integrator])
        std_diffs = np.array(energy_diff_std[integrator])

        # Remove NaN entries for plotting
        valid = ~np.isnan(mean_diffs) & ~np.isnan(std_diffs)
        dt_plot = np.array(dt_values)[valid]
        mean_plot = mean_diffs[valid]
        std_plot = std_diffs[valid]

        # Plot on log-log scale with error bars
        ax.errorbar(
            dt_plot,
            mean_plot,
            yerr=std_plot,
            marker='o',
            label=integrator,
            linewidth=2,
            markersize=6,
            capsize=4
        )

    # Plot the machine epsilon as a horizontal dashed line
    ax.axhline(y=machine_epsilon, color='red', linestyle='--', linewidth=1.5, label=r'\textbf{Machine Precision}')

    # Annotate the machine epsilon line using axes coordinates for visibility
    ax.text(
        0.8,  # x position (80% from the left)
        0.15,  # y position (15% from the bottom)
        r'$\epsilon_{\mathrm{machine}} \approx 2.22 \times 10^{-16}$',
        color='red',
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left',
        transform=ax.transAxes  # Use axes coordinates
    )

    # Use LaTeX for labels and title
    ax.set_xlabel(r'\textbf{Time Step} ($dt$) \textbf{(dimensionless)}', fontsize=14)
    ax.set_ylabel(r'\textbf{Absolute Energy Difference} ($|E(t_f) - E(t_i)|$) \textbf{(dimensionless)}', fontsize=14)
    ax.set_title(r'\textbf{Evolution of Absolute Energy Difference with Time Step for Each Integrator}', fontsize=16)

    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Set log-log scale for better visualization
    ax.set_yscale('log')
    ax.set_xscale('log')

    # Enable minor ticks
    ax.minorticks_on()

    # Adjust layout manually to accommodate annotations
    plt.subplots_adjust(left=0.1, right=0.95, top=1, bottom=0.1)

    # Save the plot with higher resolution
    # Ensure that the results directory exists
    results_dir = 'result'  # Define your results directory
    os.makedirs(results_dir, exist_ok=True)
    energy_diff_plot_path = os.path.join(results_dir, 'energy_difference_vs_dt.png')
    plt.savefig(energy_diff_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Energy Difference vs. Time Step plot saved to '{energy_diff_plot_path}'.")

    # Optional: Display the plot
    # plt.show()

    logging.info("Energy Difference vs. Time Step Analysis Completed.")

if __name__ == '__main__':
    main()
