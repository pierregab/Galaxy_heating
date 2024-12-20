# simulation.py

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from system import System
from galaxy import Galaxy
from integrators import Integrator
import os  # Import the os module for directory operations
import ffmpeg
import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Simulation(System):
    """
    Simulation class to set up and run the simulation.

    Methods:
        run(): Run the simulation.
        plot_energy_error(): Plot the energy error over time.
        plot_angular_momentum_error(): Plot the angular momentum error over time.
        plot_trajectories(): Plot the orbit trajectories.
        plot_execution_time(): Plot execution times per step.
        compute_velocity_dispersions(): Compute and compare velocity dispersions.
        plot_velocity_histograms(): Plot histograms of initial and final velocity distributions.
        log_integrator_differences(): Compute and log differences between integrators.
    """

    def __init__(self, galaxy:Galaxy, dt:float=0.05, t_max:float=250.0, integrators:list[str]=['Leapfrog', 'RK4', 'Yoshida'], paralellised = True) -> None:
        """
        Initialize the Simulation.

        Parameters:
            galaxy (Galaxy): The galaxy instance.
            dt (float): Time step (dimensionless).
            t_max (float): Total simulation time (dimensionless).
            integrators (list): List of integrators to run. Options: 'Leapfrog', 'RK4', 'Yoshida'.
        """
        super().__init__(self.__class__.__name__)
        self.paralellised = paralellised
        self.galaxy = galaxy
        self.dt = dt
        self.t_max = t_max
        self.steps = int(self.t_max / self.dt)
        self.times = np.linspace(0, self.t_max, self.steps)
        self.integrator = Integrator()
        self.positions = {}
        self.velocities = {}
        self.energies = {}
        self.angular_momenta = {}
        self.angular_momenta_BH = {}
        self.execution_times = {}
        self.energies_BH = {}  # Dictionary to store perturber's energy for each integrator
        self.perturbers_positions = {}
        self.perturbers_velocities = {}
        self.total_energy = {}
        self.energy_error = {}

        # Validate integrators
        valid_integrators = ['Leapfrog', 'RK4', 'Yoshida']  # Added 'Yoshida' to valid integrators
        for integrator in integrators:
            if integrator not in valid_integrators:
                logging.error(f"Invalid integrator selected: {integrator}. Choose from {valid_integrators}.")
                raise ValueError(f"Invalid integrator selected: {integrator}. Choose from {valid_integrators}.")
        self.integrators = integrators

        # Create the 'result' directory if it does not exist
        self.results_dir = 'result'
        try:
            os.makedirs(self.results_dir, exist_ok=True)  # Ensure directory exists
            logging.info(f"Directory '{self.results_dir}' is ready for saving results.")
        except Exception as e:
            logging.error(f"Failed to create directory '{self.results_dir}': {e}")
            raise

        # Log simulation time parameters
        logging.info(f"Simulation time parameters:")
        logging.info(f"  Time step (dt): {self.dt} (dimensionless)")
        logging.info(f"  Total simulation time (t_max): {self.t_max} (dimensionless)")
        logging.info(f"  Number of steps: {self.steps}")

    def reset_system(self) -> "Simulation":
        """
        Reset all particles and the perturbers to their initial conditions.
        """
        logging.info("Resetting system to initial conditions.")
        # Reset all particles
        for particle in self.galaxy.particles:
            particle.reset()
        
        # Reset the perturbers if they exist
        if hasattr(self.galaxy, 'perturbers') and len(self.galaxy.perturbers):
            for pert in self.galaxy.perturbers:
                pert.reset()
        return self

    def run(self) -> None:
        """
        Run the simulation using the selected integrators in parallel.
        """
        logging.info("Starting the simulation with parallel integrator execution.")

        paralellised = self.paralellised

        # Prepare a list to hold the deep copies for each integrator
        integrator_tasks = []
        for integrator_name in self.integrators:
            # Reset the system before each integrator run
            self.reset_system()
            # Deep copy the galaxy to ensure each integrator works with the same initial conditions
            galaxy_copy = deepcopy(self.galaxy)
            integrator_tasks.append((integrator_name, galaxy_copy, self.dt, self.steps))

        # Use ProcessPoolExecutor to run integrators in parallel and if parallelised is False, run them sequentially
        if paralellised:
            with ProcessPoolExecutor(max_workers=len(self.integrators)) as executor:
                # Submit all integrator tasks
                futures = [
                    executor.submit(run_single_integrator, integrator_name, galaxy_copy, dt, steps)
                    for (integrator_name, galaxy_copy, dt, steps) in integrator_tasks
                ]

                # As each future completes, collect the results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        integrator_name = result['integrator_name']

                        # Store the execution time if available
                        if 'execution_time' in result:
                            self.execution_times[integrator_name] = result['execution_time'] * 1e3  # Convert to milliseconds

                        # Store the results
                        self.positions[integrator_name] = result['positions']
                        self.velocities[integrator_name] = result['velocities']
                        self.energies[integrator_name] = result['energies']
                        self.angular_momenta[integrator_name] = result['angular_momenta']
                        self.energies_BH[integrator_name] = result['energies_BH']
                        self.angular_momenta_BH[integrator_name] = result['angular_momenta_BH']
                        self.total_energy[integrator_name] = result['total_energy']
                        self.energy_error[integrator_name] = result['energy_error']
                        self.perturbers_positions[integrator_name] = result['positions_BH']
                        self.perturbers_velocities[integrator_name] = result['velocities_BH']

                        logging.info(f"Integrator '{integrator_name}' completed successfully.")

                    except Exception as e:
                        logging.error(f"Integrator run failed: {e}")

        else:
            try:
                for integrator_name, galaxy_copy, dt, steps in integrator_tasks:
                    result = run_single_integrator(integrator_name, galaxy_copy, dt, steps)
                    integrator_name = result['integrator_name']

                    # Store the execution time if available
                    if 'execution_time' in result:
                        self.execution_times[integrator_name] = result['execution_time'] * 1e3

                    # Store the results
                    self.positions[integrator_name] = result['positions']
                    self.velocities[integrator_name] = result['velocities']
                    self.energies[integrator_name] = result['energies']
                    self.angular_momenta[integrator_name] = result['angular_momenta']
                    self.energies_BH[integrator_name] = result['energies_BH']
                    self.angular_momenta_BH[integrator_name] = result['angular_momenta_BH']
                    self.total_energy[integrator_name] = result['total_energy']
                    self.energy_error[integrator_name] = result['energy_error']
                    self.perturbers_positions[integrator_name] = result['positions_BH']
                    self.perturbers_velocities[integrator_name] = result['velocities_BH']

                    logging.info(f"Integrator '{integrator_name}' completed successfully.")

            except Exception as e:
                logging.error(f"Integrator run failed: {e}")

        logging.info("All integrators have been executed in parallel.")


    def plot_trajectories(self, subset:int=100) -> None:
        """
        Plot the orbit trajectories in the xy-plane and xz-plane separately for each integrator.

        Parameters:
            subset (int): Number of stars to plot for clarity. Defaults to 100.
        """
        logging.info("Generating orbit trajectory plots.")

        N = len(self.galaxy.particles)
        subset = min(subset, N)  # Ensure subset does not exceed total number of stars
        indices = np.random.choice(N, subset, replace=False)  # Randomly select stars to plot

        for integrator_name in self.integrators:
            pos = self.positions[integrator_name]
            plt.figure(figsize=(12, 6))

            # x-y plot
            plt.subplot(1, 2, 1)
            for i in indices:
                plt.plot(pos[:, i, 0] * self.length_scale,
                        pos[:, i, 1] * self.length_scale,
                        linewidth=0.5, alpha=0.7)
            plt.xlabel('x (kpc)', fontsize=12)
            plt.ylabel('y (kpc)', fontsize=12)
            plt.title(f'{integrator_name}: Orbit Trajectories in x-y Plane', fontsize=14)
            plt.grid(True)
            plt.axis('equal')

            # x-z plot
            plt.subplot(1, 2, 2)
            for i in indices:
                plt.plot(pos[:, i, 0] * self.length_scale,
                        pos[:, i, 2] * self.length_scale,
                        linewidth=0.5, alpha=0.7)
            plt.xlabel('x (kpc)', fontsize=12)
            plt.ylabel('z (kpc)', fontsize=12)
            plt.title(f'{integrator_name}: Orbit Trajectories in x-z Plane', fontsize=14)
            plt.grid(True)
            plt.axis('equal')

            # Plot the perturber's trajectory
            if integrator_name in self.perturbers_positions:
                # Check if there is a perturber
                if hasattr(self.galaxy, 'perturbers') and len(self.galaxy.perturbers):
                    pos_BH = self.perturbers_positions[integrator_name]  # [P, steps, 3]
                    for pertIndex in range(pos_BH.shape[0]):
                        # x-y plot
                        plt.subplot(1, 2, 1)
                        plt.plot(pos_BH[pertIndex, :, 0] * self.length_scale,
                                pos_BH[pertIndex, :, 1] * self.length_scale,
                                color='red', linestyle=['-', '--', '-.', ':'][pertIndex%4], linewidth=2, label=f'Perturber {pertIndex+1}')
                        plt.legend()

                        # x-z plot
                        plt.subplot(1, 2, 2)
                        plt.plot(pos_BH[pertIndex, :, 0] * self.length_scale,
                                pos_BH[pertIndex, :, 2] * self.length_scale,
                                color='red', linestyle=['-', '--', '-.', ':'][pertIndex%4], linewidth=2, label=f'Perturber {pertIndex+1}')
                        plt.legend()

            plt.tight_layout()
            filename = f'orbit_{integrator_name.lower()}.png'
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()
            logging.info(f"{integrator_name} orbit trajectories plots saved to '{self.results_dir}/{filename}'.")

    def plot_energy_error(self) -> None:
        """
        Plot the total energy conservation error over time for each integrator.
        """
        logging.info("Generating total energy conservation error plot.")

        plt.figure(figsize=(12, 8))

        for integrator_name in self.integrators:
            if integrator_name not in self.total_energy or self.total_energy[integrator_name] is None:
                logging.warning(f"Total energy data for integrator '{integrator_name}' is unavailable. Skipping.")
                continue

            # Time array in physical units
            times_physical = self.times * self.time_scale  # Time in Myr

            # Total energy from simulation
            total_E = self.total_energy[integrator_name]  # [steps]

            # Initial total energy
            E_initial = total_E[0]

            # Compute relative energy error
            relative_E_error = (total_E - E_initial) / np.abs(E_initial)  # [steps]

            # Plot the relative energy error
            plt.plot(
                times_physical,
                np.abs(relative_E_error),  # Taking absolute value for better visualization
                label=f"{integrator_name}",
                linewidth=1
            )

        plt.xlabel('Time (Myr)', fontsize=14)
        plt.ylabel('Relative Total Energy Error', fontsize=14)
        plt.title('Total Energy Conservation Error Over Time', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')  # Using logarithmic scale to capture small errors
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'total_energy_error.png'))
        plt.close()
        logging.info(f"Total energy conservation error plot saved to '{self.results_dir}/total_energy_error.png'.")


    def plot_angular_momentum_error(self) -> None:
        """
        Plot the angular momentum conservation error over time, including a dedicated subplot for Lz.
        """
        logging.info("Generating angular momentum conservation error plot.")
        
        plt.figure(figsize=(14, 15))  # Increased height to accommodate an extra subplot

        for integrator_name in self.integrators:
            # Time array in physical units (e.g., Myr)
            times_physical = self.times * self.time_scale  # Assuming self.times is in simulation units

            if len(self.galaxy.perturbers):
                # Total angular momentum for stars and perturbers
                L_stars = np.sum(self.angular_momenta[integrator_name], axis=1)  # [steps, 3]
                L_BH = np.sum(self.angular_momenta_BH[integrator_name], axis=1) if self.angular_momenta_BH[integrator_name] is not None else 0
                L_total = L_stars + L_BH  # [steps, 3]
                L0_total = L_total[0]
            else:
                # Total angular momentum for stars only
                L_total = np.sum(self.angular_momenta[integrator_name], axis=1)  # [steps, 3]
                L0_total = L_total[0]

            # Compute the magnitude of total angular momentum at each step
            L_magnitude = np.linalg.norm(L_total, axis=1)  # [steps]
            L0_magnitude = np.linalg.norm(L0_total)

            # Compute relative error in magnitude
            relative_error_magnitude = np.abs(L_magnitude - L0_magnitude) / np.abs(L0_magnitude)  # [steps]

            # Compute relative error for each component
            relative_error_components = np.abs(L_total - L0_total) / np.abs(L0_total)  # [steps, 3]
            relative_error_x = relative_error_components[:, 0]
            relative_error_y = relative_error_components[:, 1]
            relative_error_z = relative_error_components[:, 2]

            # Plot Relative Error in Total Angular Momentum Magnitude
            plt.subplot(3, 1, 1)  # Changed from (2,1,1) to (3,1,1)
            plt.plot(times_physical, relative_error_magnitude, label=integrator_name, linewidth=1)
            plt.xlabel('Time (Myr)', fontsize=14)
            plt.ylabel('Relative Error in |L|', fontsize=14)
            plt.title('Total Angular Momentum Conservation Error', fontsize=16)
            plt.yscale('log')
            plt.grid(True)

            # Plot Relative Error in Each Angular Momentum Component
            plt.subplot(3, 1, 2)  # Changed from (2,1,2) to (3,1,2)
            plt.plot(times_physical, relative_error_x, label=f'{integrator_name} Lx', linewidth=1)
            plt.plot(times_physical, relative_error_y, label=f'{integrator_name} Ly', linewidth=1)
            plt.plot(times_physical, relative_error_z, label=f'{integrator_name} Lz', linewidth=1)
            plt.xlabel('Time (Myr)', fontsize=14)
            plt.ylabel('Relative Error in L Components', fontsize=14)
            plt.title('Angular Momentum Conservation Error per Component', fontsize=16)
            plt.yscale('log')
            plt.legend(fontsize=12)
            plt.grid(True)

            # Add a Dedicated Subplot for Lz
            plt.subplot(3, 1, 3)
            plt.plot(times_physical, relative_error_z, label=f'{integrator_name} Lz', linewidth=1)
            plt.xlabel('Time (Myr)', fontsize=14)
            plt.ylabel('Relative Error in Lz', fontsize=14)
            plt.title('Angular Momentum Conservation Error for Lz Component', fontsize=16)
            plt.yscale('log')
            plt.legend(fontsize=12)
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'angular_momentum_error.png'))
        plt.close()
        logging.info(f"Angular momentum conservation error plot saved to '{self.results_dir}/angular_momentum_error.png'.")


    def plot_execution_time(self) -> None:
        """
        Plot execution times per step for the integrators.
        """
        logging.info("Generating execution time comparison plot.")

        plt.figure(figsize=(10, 8))
        methods = list(self.execution_times.keys())
        times_exec = [self.execution_times[method] for method in methods]  # in ms
        plt.bar(methods, times_exec, color=['blue', 'orange'])
        plt.ylabel('Average Time per Step (ms)', fontsize=14)
        plt.title('Integrator Execution Time Comparison', fontsize=16)
        for i, v in enumerate(times_exec):
            plt.text(i, v + 0.05 * max(times_exec), f"{v:.3f} ms", ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'execution_times.png'))
        plt.close()
        logging.info(f"Execution time comparison plot saved to '{self.results_dir}/execution_times.png'.")

    def compute_velocity_dispersions(self) -> None:
        """
        Compute the radial and vertical velocity dispersions at specific moments
        (start, 1/3, 2/3, final) after integration and compare them to the initial values.
        Includes error bars representing uncertainties in the fitted dispersions.
        Plots are in real physical units (kpc for radius and km/s for velocity dispersion).
        """
        logging.info("Computing velocity dispersions at specific moments after integration.")

        # Define the time steps: start, 1/3, 2/3, and final
        moments = {
            'Start': 0,
            '1/3': self.steps // 3,
            '2/3': 2 * self.steps // 3,
            'Final': self.steps - 1
        }

        # Define the order of moments for consistent coloring
        moment_order = list(moments.keys())  # ['Start', '1/3', '2/3', 'Final']
        n_moments = len(moment_order)

        # Choose a blue-to-red colormap
        cmap = plt.get_cmap('coolwarm')  # 'coolwarm' transitions from blue to red

        # Generate colors for each moment based on their order
        colors = [cmap(i / (n_moments - 1)) for i in range(n_moments)]

        # Define markers for sigma_R and sigma_z
        markers_R = ['o', '^', 's', 'D']  # Circle, Triangle Up, Square, Diamond
        markers_z = ['s', 'v', 'D', 'x']  # Square, Triangle Down, Diamond, Cross

        # Create a dictionary to hold styles for each moment
        moment_styles = {
            moment: {
                'color': color,
                'marker_R': markers_R[i],
                'marker_z': markers_z[i],
                'linestyle_R': '-',    # Solid line for sigma_R
                'linestyle_z': '--'    # Dashed line for sigma_z
            }
            for i, (moment, color) in enumerate(zip(moment_order, colors))
        }

        # Define radial bins in physical units
        R_c_bins = np.linspace(np.min(self.galaxy.R_c) * self.length_scale, 
                            np.max(self.galaxy.R_c) * self.length_scale, 10)
        indices = np.digitize(self.galaxy.R_c * self.length_scale, R_c_bins)
        R_c_centers = 0.5 * (R_c_bins[:-1] + R_c_bins[1:])

        # Compute initial sigma_R_init and sigma_z_init in physical units
        sigma_R_init = []
        sigma_z_init = []
        for i in range(1, len(R_c_bins)):
            idx = np.where(indices == i)[0]
            if len(idx) > 1:
                # Initial sigma values (from theoretical expressions) converted to km/s
                sigma_R_initial_val = np.mean(self.galaxy.initial_sigma_R[idx]) * self.velocity_scale_kms
                sigma_z_initial_val = np.mean(self.galaxy.initial_sigma_z[idx]) * self.velocity_scale_kms
                sigma_R_init.append(sigma_R_initial_val)
                sigma_z_init.append(sigma_z_initial_val)
            else:
                # Not enough stars to compute dispersion
                sigma_R_init.append(np.nan)
                sigma_z_init.append(np.nan)

        for integrator_name in self.integrators:
            logging.info(f"Processing dispersions for integrator: {integrator_name}")

            # Initialize dictionaries to store dispersions and uncertainties
            dispersions_R = {moment: [] for moment in moments}
            dispersions_z = {moment: [] for moment in moments}
            uncertainties_R = {moment: [] for moment in moments}
            uncertainties_z = {moment: [] for moment in moments}

            for moment_label, step_idx in moments.items():
                logging.info(f"  Computing dispersions at moment: {moment_label} (Step {step_idx})")

                # Extract positions and velocities at the specified step
                pos = self.positions[integrator_name][step_idx]  # [N, 3]
                vel = self.velocities[integrator_name][step_idx]  # [N, 3]

                # Compute cylindrical coordinates
                x = pos[:, 0] * self.length_scale  # Convert to kpc
                y = pos[:, 1] * self.length_scale  # Convert to kpc
                z = pos[:, 2] * self.length_scale  # Convert to kpc
                R = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)

                # Compute velocities in cylindrical coordinates and convert to km/s
                v_x = vel[:, 0] * self.velocity_scale_kms  # Convert to km/s
                v_y = vel[:, 1] * self.velocity_scale_kms  # Convert to km/s
                v_z = vel[:, 2] * self.velocity_scale_kms  # Convert to km/s

                # Compute radial and azimuthal velocities
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_R_final = (x * v_x + y * v_y) / R
                    v_phi_final = (x * v_y - y * v_x) / R

                # Handle division by zero for R=0
                v_R_final = np.nan_to_num(v_R_final)
                v_phi_final = np.nan_to_num(v_phi_final)

                # Compute residual velocities
                delta_v_R = v_R_final
                delta_v_z = v_z

                # Initialize lists for this moment
                sigma_R_fit = []
                sigma_z_fit = []
                sigma_R_unc_fit = []
                sigma_z_unc_fit = []

                for i in range(1, len(R_c_bins)):
                    idx = np.where(indices == i)[0]
                    if len(idx) > 1:
                        # Residual velocities for this bin
                        delta_v_R_bin = delta_v_R[idx]
                        delta_v_z_bin = delta_v_z[idx]

                        # Fit Gaussian to the radial residual velocities
                        params_R = norm.fit(delta_v_R_bin)
                        sigma_R = params_R[1]  # Standard deviation
                        sigma_R_fit.append(sigma_R)

                        # Fit Gaussian to the vertical residual velocities
                        params_z = norm.fit(delta_v_z_bin)
                        sigma_z = params_z[1]
                        sigma_z_fit.append(sigma_z)

                        # Compute uncertainties in the fitted dispersions
                        N_bin = len(idx)
                        sigma_R_unc = sigma_R / np.sqrt(2 * (N_bin - 1))
                        sigma_z_unc = sigma_z / np.sqrt(2 * (N_bin - 1))
                        sigma_R_unc_fit.append(sigma_R_unc)
                        sigma_z_unc_fit.append(sigma_z_unc)
                    else:
                        # Not enough stars to compute dispersion
                        sigma_R_fit.append(np.nan)
                        sigma_z_fit.append(np.nan)
                        sigma_R_unc_fit.append(np.nan)
                        sigma_z_unc_fit.append(np.nan)

                # Store the dispersions and uncertainties
                dispersions_R[moment_label] = sigma_R_fit
                dispersions_z[moment_label] = sigma_z_fit
                uncertainties_R[moment_label] = sigma_R_unc_fit
                uncertainties_z[moment_label] = sigma_z_unc_fit

            # Plot the dispersions with error bars and lines connecting points
            plt.figure(figsize=(14, 8))
            for moment_label in moments:
                if not np.all(np.isnan(dispersions_R[moment_label])):
                    # Plot sigma_R
                    plt.errorbar(
                        R_c_centers,
                        np.array(dispersions_R[moment_label]),
                        yerr=np.array(uncertainties_R[moment_label]),
                        marker=moment_styles[moment_label]['marker_R'],
                        linestyle=moment_styles[moment_label]['linestyle_R'],
                        label=f"{moment_label} σ_R",
                        color=moment_styles[moment_label]['color'],
                        capsize=3,
                        markersize=6,
                        linewidth=1.5
                    )
                    # Plot sigma_z
                    plt.errorbar(
                        R_c_centers,
                        np.array(dispersions_z[moment_label]),
                        yerr=np.array(uncertainties_z[moment_label]),
                        marker=moment_styles[moment_label]['marker_z'],
                        linestyle=moment_styles[moment_label]['linestyle_z'],
                        label=f"{moment_label} σ_z",
                        color=moment_styles[moment_label]['color'],
                        capsize=3,
                        markersize=6,
                        linewidth=1.5
                    )

            # Plot initial theoretical dispersions as solid lines
            sigma_R_init = np.array(sigma_R_init)
            sigma_z_init = np.array(sigma_z_init)
            plt.plot(R_c_centers, sigma_R_init, 'k-', label='Initial σ_R (Theoretical)', linewidth=2)
            plt.plot(R_c_centers, sigma_z_init, 'k--', label='Initial σ_z (Theoretical)', linewidth=2)

            plt.xlabel('Reference Radius $R_c$ (kpc)', fontsize=16)
            plt.ylabel('Velocity Dispersion σ (km/s)', fontsize=16)
            plt.title(f'Velocity Dispersions at Different Moments ({integrator_name})', fontsize=18)

            # Place the legend outside the plot to avoid overlapping with data
            plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect to make space for the legend

            filename = f'velocity_dispersions_{integrator_name.lower()}_moments_physical.png'
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()
            logging.info(f"Velocity dispersion comparison plot with moments for {integrator_name} saved to '{self.results_dir}/{filename}'.")

            # Optionally, print the dispersions
            for i in range(len(R_c_centers)):
                sigma_R_init_val = sigma_R_init[i]
                sigma_z_init_val = sigma_z_init[i]
                for moment_label in moments:
                    # Ensure the lists have the correct length
                    if i < len(dispersions_R[moment_label]):
                        sigma_R = dispersions_R[moment_label][i]
                        sigma_z = dispersions_z[moment_label][i]
                        sigma_R_unc = uncertainties_R[moment_label][i]
                        sigma_z_unc = uncertainties_z[moment_label][i]
                    else:
                        sigma_R = np.nan
                        sigma_z = np.nan
                        sigma_R_unc = np.nan
                        sigma_z_unc = np.nan
                    logging.info(f"{integrator_name} - Moment: {moment_label}, R_c = {R_c_centers[i]:.2f} kpc: "
                                f"σ_R = {sigma_R:.2f} ± {sigma_R_unc:.2f} km/s, "
                                f"σ_z = {sigma_z:.2f} ± {sigma_z_unc:.2f} km/s, "
                                f"Initial σ_R = {sigma_R_init_val:.2f} km/s, Initial σ_z = {sigma_z_init_val:.2f} km/s")

    def compute_velocity_dispersions_continuous(self) -> None:
        """
        Compute the radial (σ_R) and vertical (σ_z) velocity dispersions over time
        and plot them as continuous curves with respect to the simulation time.
        Utilizes a color gradient to highlight the increase in velocity dispersion over time.
        Plots are in real physical units (kpc for radius and km/s for velocity dispersion).
        """
        logging.info("Computing continuous velocity dispersions over time and radius.")

        # Define radial bins in physical units (kpc)
        R_min = np.min(self.galaxy.R_c) * self.length_scale
        R_max = np.max(self.galaxy.R_c) * self.length_scale
        num_bins = 100  # Number of radial bins
        R_c_bins = np.linspace(R_min, R_max, num_bins + 1)  # (num_bins +1,)
        R_c_centers = 0.5 * (R_c_bins[:-1] + R_c_bins[1:])  # (num_bins,)
        indices = np.digitize(self.galaxy.R_c * self.length_scale, R_c_bins)  # (N,)

        # Define time sampling parameters
        max_samples = 1000  # Maximum number of samples to plot for clarity
        sample_interval = max(1, self.steps // max_samples)  # Adjust sample interval based on steps
        sampled_steps = np.arange(0, self.steps, sample_interval)
        if (self.steps - 1) not in sampled_steps:
            sampled_steps = np.append(sampled_steps, self.steps - 1)  # Ensure final step is included

        # Initialize dictionaries to store σ_R and σ_z over time and radius
        sigma_R_time = {integrator: [] for integrator in self.integrators}
        sigma_z_time = {integrator: [] for integrator in self.integrators}
        time_values = {integrator: [] for integrator in self.integrators}

        # Loop over integrators and sampled steps to compute dispersions
        for integrator_name in self.integrators:
            logging.info(f"Processing dispersions for integrator: {integrator_name}")
            for step_idx in sampled_steps:
                # Extract positions and velocities at the step
                pos = self.positions[integrator_name][step_idx]  # [N, 3]
                vel = self.velocities[integrator_name][step_idx]  # [N, 3]

                # Convert positions to physical units (kpc)
                x = pos[:, 0] * self.length_scale
                y = pos[:, 1] * self.length_scale
                z = pos[:, 2] * self.length_scale
                R = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)

                # Convert velocities to physical units (km/s)
                v_x = vel[:, 0] * self.velocity_scale_kms
                v_y = vel[:, 1] * self.velocity_scale_kms
                v_z = vel[:, 2] * self.velocity_scale_kms

                # Compute radial and azimuthal velocities
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_R = (x * v_x + y * v_y) / R
                    v_phi = (x * v_y - y * v_x) / R

                # Handle division by zero for R=0
                v_R = np.nan_to_num(v_R)
                v_phi = np.nan_to_num(v_phi)

                # Compute residual velocities
                delta_v_R = v_R
                delta_v_z = v_z

                # Compute dispersions in each radial bin
                for j in range(1, len(R_c_bins)):
                    idx = np.where(indices == j)[0]
                    if len(idx) > 1:
                        # Radial velocity dispersion (σ_R)
                        sigma_R = np.std(delta_v_R[idx])
                        sigma_R_time[integrator_name].append(sigma_R)  # Already in km/s

                        # Vertical velocity dispersion (σ_z)
                        sigma_z = np.std(delta_v_z[idx])
                        sigma_z_time[integrator_name].append(sigma_z)  # Already in km/s
                    else:
                        sigma_R_time[integrator_name].append(np.nan)
                        sigma_z_time[integrator_name].append(np.nan)

                # Record the current time in Myr
                current_time = step_idx * self.dt * self.time_scale  # Convert to Myr
                time_values[integrator_name].extend([current_time] * num_bins)

        # Ensure the 'results_dir' exists
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            logging.info(f"Created directory '{self.results_dir}' for saving plots.")

        # Function to plot σ vs R_c with color gradient representing time
        def plot_sigma_vs_Rc(integrator_name, sigma_time, time_vals, sigma_label, cmap_name, filename):
            """
            Plot σ vs R_c curves colored by time.

            Parameters:
                integrator_name (str): Name of the integrator.
                sigma_time (list of floats): Velocity dispersions at sampled steps and radial bins.
                time_vals (list of floats): Corresponding simulation times for each σ value.
                sigma_label (str): Label for the velocity dispersion (σ_R or σ_z).
                cmap_name (str): Colormap name.
                filename (str): Filename to save the plot.
            """
            fig, ax = plt.subplots(figsize=(14, 8))

            # Number of sampled steps
            num_samples = len(sampled_steps)

            # Number of radial bins
            num_bins = len(R_c_centers)

            # Reshape sigma_time and time_vals
            sigma_array = np.array(sigma_time).reshape(num_samples, num_bins)  # [samples, bins]
            time_array = np.array(time_vals).reshape(num_samples, num_bins)    # [samples, bins]

            # Create a colormap and normalization based on time
            cmap = plt.get_cmap(cmap_name)
            norm = Normalize(vmin=0, vmax=self.t_max * self.time_scale)  # Normalize to physical time (Myr)

            # Collect all lines and their corresponding times
            lines = []
            colors = []
            for i in range(num_samples):
                sigma_vals = sigma_array[i]
                R_vals = R_c_centers
                # Create (R, σ) pairs, ignoring NaNs
                valid = ~np.isnan(sigma_vals)
                if np.any(valid):
                    line = np.stack([R_vals[valid], sigma_vals[valid]], axis=1)
                    lines.append(line)
                    # Assign the time of this step to the color
                    colors.append(time_array[i][valid][0])  # Assuming all valid σ have the same time

            # Create LineCollection
            lc = LineCollection(lines, cmap=cmap, norm=norm, linewidth=2, alpha=0.8)
            lc.set_array(np.array(colors))
            lc.set_linewidth(2)

            # Add LineCollection to the axes
            ax.add_collection(lc)

            # Set limits
            ax.set_xlim(R_min, R_max)
            sigma_max = np.nanmax(sigma_array)
            ax.set_ylim(0, sigma_max * 1.1)  # Slight padding

            # Add colorbar
            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label('Time (Myr)', fontsize=14)

            # Set labels and title
            ax.set_xlabel('Radius $R_c$ (kpc)', fontsize=16)
            ax.set_ylabel(f'{sigma_label} (km/s)', fontsize=16)
            ax.set_title(f'{sigma_label} vs Radius $R_c$ Over Time ({integrator_name})', fontsize=18)

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.5)

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close(fig)
            logging.info(f"{sigma_label} vs R_c plot for {integrator_name} saved to '{self.results_dir}/{filename}'.")

        # Plot σ_R
        for integrator_name in self.integrators:
            logging.info(f"Plotting σ_R vs R_c for {integrator_name}")
            plot_sigma_vs_Rc(
                integrator_name=integrator_name,
                sigma_time=sigma_R_time[integrator_name],
                time_vals=time_values[integrator_name],
                sigma_label='Radial Velocity Dispersion σ_R',
                cmap_name='viridis',
                filename=f'velocity_dispersion_sigmaR_time_gradient_{integrator_name.lower()}.png'
            )

        # Plot σ_z
        for integrator_name in self.integrators:
            logging.info(f"Plotting σ_z vs R_c for {integrator_name}")
            plot_sigma_vs_Rc(
                integrator_name=integrator_name,
                sigma_time=sigma_z_time[integrator_name],
                time_vals=time_values[integrator_name],
                sigma_label='Vertical Velocity Dispersion σ_z',
                cmap_name='plasma',
                filename=f'velocity_dispersion_sigmaZ_time_gradient_{integrator_name.lower()}.png'
            )

    def plot_velocity_histograms(self, subset:int=200) -> None:
        """
        Plot histograms of initial and final velocity distributions.

        Parameters:
            subset (int): Number of stars to plot for clarity. Defaults to 200.
        """
        logging.info("Generating velocity histograms.")

        N = len(self.galaxy.particles)
        subset = min(subset, N)  # Ensure subset does not exceed total number of stars
        indices = np.random.choice(N, subset, replace=False)  # Randomly select stars to plot

        for integrator_name in self.integrators:
            # Initial velocities
            initial_vx = self.galaxy.initial_velocities[indices, 0]
            initial_vy = self.galaxy.initial_velocities[indices, 1]
            initial_vz = self.galaxy.initial_velocities[indices, 2]
            initial_speed = np.linalg.norm(self.galaxy.initial_velocities[indices], axis=1)

            # Final velocities from the integrator
            final_vx = self.velocities[integrator_name][-1][indices, 0]
            final_vy = self.velocities[integrator_name][-1][indices, 1]
            final_vz = self.velocities[integrator_name][-1][indices, 2]
            final_speed = np.linalg.norm(self.velocities[integrator_name][-1][indices], axis=1)

            # Plot histograms for v_R, v_phi, v_z
            plt.figure(figsize=(18, 6))

            # Radial Velocity
            plt.subplot(1, 3, 1)
            plt.hist(initial_vx, bins=30, alpha=0.5, label='Initial $v_x$', color='blue', density=True)
            plt.hist(final_vx, bins=30, alpha=0.5, label='Final $v_x$', color='green', density=True)
            plt.xlabel('$v_x$ (dimensionless)', fontsize=12)
            plt.ylabel('Normalized Frequency', fontsize=12)
            plt.title(f'Radial Velocity Distribution ({integrator_name})', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            # Azimuthal Velocity
            plt.subplot(1, 3, 2)
            plt.hist(initial_vy, bins=30, alpha=0.5, label='Initial $v_y$', color='blue', density=True)
            plt.hist(final_vy, bins=30, alpha=0.5, label='Final $v_y$', color='green', density=True)
            plt.xlabel('$v_y$ (dimensionless)', fontsize=12)
            plt.ylabel('Normalized Frequency', fontsize=12)
            plt.title(f'Azimuthal Velocity Distribution ({integrator_name})', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            # Vertical Velocity
            plt.subplot(1, 3, 3)
            plt.hist(initial_vz, bins=30, alpha=0.5, label='Initial $v_z$', color='blue', density=True)
            plt.hist(final_vz, bins=30, alpha=0.5, label='Final $v_z$', color='green', density=True)
            plt.xlabel('$v_z$ (dimensionless)', fontsize=12)
            plt.ylabel('Normalized Frequency', fontsize=12)
            plt.title(f'Vertical Velocity Distribution ({integrator_name})', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            plt.tight_layout()
            filename = f'velocity_histograms_{integrator_name.lower()}.png'
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()
            logging.info(f"Velocity histograms for {integrator_name} saved to '{self.results_dir}/{filename}'.")

            # Plot histogram for total speed
            plt.figure(figsize=(12, 6))
            plt.hist(initial_speed, bins=30, alpha=0.5, label='Initial Speed', color='blue', density=True)
            plt.hist(final_speed, bins=30, alpha=0.5, label='Final Speed', color='green', density=True)
            plt.xlabel('Speed (dimensionless)', fontsize=12)
            plt.ylabel('Normalized Frequency', fontsize=12)
            plt.title(f'Total Speed Distribution ({integrator_name})', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            filename_speed = f'total_speed_histogram_{integrator_name.lower()}.png'
            plt.savefig(os.path.join(self.results_dir, filename_speed))
            plt.close()
            logging.info(f"Total speed histogram for {integrator_name} saved to '{self.results_dir}/{filename_speed}'.")

    def log_integrator_differences(self) -> None:
        """
        Compute and log the differences between RK4 and Leapfrog integrators for stars and the perturbers.
        """
        logging.info("Computing differences between RK4 and Leapfrog integrators.")

        # Check if both integrators were run
        required_integrators = ['RK4', 'Leapfrog']
        if not all(impl in self.integrators for impl in required_integrators):
            logging.warning("Both RK4 and Leapfrog integrators must be selected to compute differences.")
            return

        # Retrieve positions and velocities from both integrators
        try:
            positions_RK4 = self.positions['RK4']            # Shape: [steps, N, 3]
            velocities_RK4 = self.velocities['RK4']          # Shape: [steps, N, 3]
            positions_Leapfrog = self.positions['Leapfrog']  # Shape: [steps, N, 3]
            velocities_Leapfrog = self.velocities['Leapfrog']# Shape: [steps, N, 3]
        except KeyError as e:
            logging.error(f"Missing integrator data: {e}")
            return

        # Ensure that both integrators have the same number of steps and particles
        if positions_RK4.shape != positions_Leapfrog.shape or velocities_RK4.shape != velocities_Leapfrog.shape:
            logging.error("Integrator results have mismatched shapes. Cannot compute differences.")
            return

        # Compute differences at the final time step for stars
        final_positions_RK4 = positions_RK4[-1]          # Shape: [N, 3]
        final_positions_Leapfrog = positions_Leapfrog[-1]# Shape: [N, 3]
        final_velocities_RK4 = velocities_RK4[-1]        # Shape: [N, 3]
        final_velocities_Leapfrog = velocities_Leapfrog[-1]  # Shape: [N, 3]

        # Compute position and velocity differences for stars
        position_diff = final_positions_RK4 - final_positions_Leapfrog  # Shape: [N, 3]
        velocity_diff = final_velocities_RK4 - final_velocities_Leapfrog  # Shape: [N, 3]

        # Compute RMS differences for stars
        position_rms = np.sqrt(np.mean(np.sum(position_diff**2, axis=1)))
        velocity_rms = np.sqrt(np.mean(np.sum(velocity_diff**2, axis=1)))

        logging.info(f"Average RMS position difference between RK4 and Leapfrog for stars: {position_rms:.6e} (dimensionless units)")
        logging.info(f"Average RMS velocity difference between RK4 and Leapfrog for stars: {velocity_rms:.6e} (dimensionless units)")

        # If perturbers data is available, compute differences for the perturbers
        if hasattr(self.galaxy, 'perturbers') and len(self.galaxy.perturbers) > 0:
            if 'RK4' in self.perturbers_positions and 'Leapfrog' in self.perturbers_positions:
                positions_BH_RK4 = self.perturbers_positions['RK4']        # Shape: [P, steps, 3]
                velocities_BH_RK4 = self.perturbers_velocities['RK4']      # Shape: [P, steps, 3]
                positions_BH_Leapfrog = self.perturbers_positions['Leapfrog']  # Shape: [P, steps, 3]
                velocities_BH_Leapfrog = self.perturbers_velocities['Leapfrog']# Shape: [P, steps, 3]

                # Ensure the shapes match
                if positions_BH_RK4.shape != positions_BH_Leapfrog.shape or velocities_BH_RK4.shape != velocities_BH_Leapfrog.shape:
                    logging.error("Perturbers' integrator results have mismatched shapes. Cannot compute differences.")
                    return

                # Compute differences at the final time step for each perturber
                final_positions_BH_RK4 = positions_BH_RK4[:, -1, :]        # Shape: [P, 3]
                final_positions_BH_Leapfrog = positions_BH_Leapfrog[:, -1, :]  # Shape: [P, 3]
                final_velocities_BH_RK4 = velocities_BH_RK4[:, -1, :]      # Shape: [P, 3]
                final_velocities_BH_Leapfrog = velocities_BH_Leapfrog[:, -1, :]  # Shape: [P, 3]

                # Compute position and velocity differences for perturbers
                position_diff_BH = final_positions_BH_RK4 - final_positions_BH_Leapfrog  # Shape: [P, 3]
                velocity_diff_BH = final_velocities_BH_RK4 - final_velocities_BH_Leapfrog  # Shape: [P, 3]

                # Compute Euclidean distances for the perturbers
                position_distance_BH = np.linalg.norm(position_diff_BH, axis=1)  # Shape: [P]
                velocity_distance_BH = np.linalg.norm(velocity_diff_BH, axis=1)  # Shape: [P]

                # Compute RMS differences for perturbers
                position_rms_BH = np.sqrt(np.mean(position_distance_BH**2))
                velocity_rms_BH = np.sqrt(np.mean(velocity_distance_BH**2))

                logging.info(f"Average RMS position difference between RK4 and Leapfrog for perturbers: {position_rms_BH:.6e} (dimensionless units)")
                logging.info(f"Average RMS velocity difference between RK4 and Leapfrog for perturbers: {velocity_rms_BH:.6e} (dimensionless units)")

                # Optional: Detailed per-perturber differences
                for idx, pert in enumerate(self.galaxy.perturbers):
                    logging.debug(f"Perturber {idx+1}: Position difference = {position_distance_BH[idx]:.6e}, Velocity difference = {velocity_distance_BH[idx]:.6e}")
            else:
                logging.warning("Perturber data is not available for both RK4 and Leapfrog integrators.")
        else:
            logging.warning("No perturbers present in the simulation. Skipping perturber differences.")

    def plot_equipotential(self) -> None:
        """
        Generate and save black-and-white (grayscale) equipotential line plots
        of the galaxy potential in the x-y and x-z planes, showing the entire galaxy.
        Both plots are included in the same figure with specified limits.
        """
        logging.info("Generating equipotential line plots in x-y and x-z planes.")

        # Define the plot limits
        x_max = 10.0  # in dimensionless units
        y_max = 10.0
        z_max = 10.0

        # Define grid resolution
        grid_points_xy = 500  # Higher for smoother contours in x-y
        grid_points_xz = 500  # Higher for smoother contours in x-z

        # --- Equipotential in x-y Plane ---
        x = np.linspace(-x_max, x_max, grid_points_xy)
        y = np.linspace(-y_max, y_max, grid_points_xy)
        X_xy, Y_xy = np.meshgrid(x, y)
        Z_xy = np.zeros_like(X_xy)  # z=0 for x-y plane

        # Compute potential at each (x, y, z=0)
        R_xy = np.sqrt(X_xy**2 + Y_xy**2)
        potential_xy = self.galaxy.potential(R_xy, Z_xy)

        # --- Equipotential in x-z Plane ---
        x = np.linspace(-x_max, x_max, grid_points_xz)
        z = np.linspace(-z_max, z_max, grid_points_xz)
        X_xz, Z_xz = np.meshgrid(x, z)
        Y_xz = np.zeros_like(X_xz)  # y=0 for x-z plane

        # Compute potential at each (x, y=0, z)
        R_xz = np.sqrt(X_xz**2 + Y_xz**2)
        potential_xz = self.galaxy.potential(R_xz, Z_xz)

        # Create a single figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 7))

        # Plot equipotential in x-y plane
        levels_xy = np.linspace(np.min(potential_xy), np.max(potential_xy), 50)
        cs_xy = axs[0].contour(X_xy * self.length_scale, Y_xy * self.length_scale, potential_xy,
                               levels=levels_xy, colors='black', linewidths=0.5)
        axs[0].set_xlabel('x (kpc)', fontsize=14)
        axs[0].set_ylabel('y (kpc)', fontsize=14)
        axs[0].set_title('Equipotential Lines in x-y Plane', fontsize=16)
        axs[0].set_xlim(-x_max * self.length_scale, x_max * self.length_scale)
        axs[0].set_ylim(-y_max * self.length_scale, y_max * self.length_scale)
        axs[0].set_aspect('equal')
        axs[0].grid(True, linestyle='--', alpha=0.5)

        # Plot equipotential in x-z plane
        levels_xz = np.linspace(np.min(potential_xz), np.max(potential_xz), 50)
        cs_xz = axs[1].contour(X_xz * self.length_scale, Z_xz * self.length_scale, potential_xz,
                               levels=levels_xz, colors='black', linewidths=0.5)
        axs[1].set_xlabel('x (kpc)', fontsize=14)
        axs[1].set_ylabel('z (kpc)', fontsize=14)
        axs[1].set_title('Equipotential Lines in x-z Plane', fontsize=16)
        axs[1].set_xlim(-x_max * self.length_scale, x_max * self.length_scale)
        axs[1].set_ylim(-z_max * self.length_scale, z_max * self.length_scale)
        axs[1].set_aspect('equal')
        axs[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        filename = 'galaxy_equipotential_xy_xz.png'
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()
        logging.info(f"Galaxy equipotential x-y and x-z plane plots saved to '{self.results_dir}/{filename}'.")

    def plot_galaxy_snapshots(self, n_snapshots:int=4, figsize:tuple[float, float]|None=None, independantFig:bool=False) -> None:
        """
        Generate and save black-and-white (grayscale) plots of the galaxy showing all stars
        and the perturbers (if any) at multiple time snapshots.
        Each snapshot includes two subplots: x-y plane and x-z plane.

        Parameters:
            n_snapshots (int): Number of snapshots to plot. Defaults to 4.
            figsize (tuple): Figure size in inches. Defaults to (20, n_snapshots * 5).
        """
        logging.info("Generating galaxy snapshots at multiple time steps.")

        n_fig = n_snapshots
        if independantFig:
            n_fig = 1
        if figsize is None:
            # Increase the figure size for better visibility
            figsize = (20,n_fig*5)

        # Define the snapshot steps (equally spaced)
        snapshot_steps = np.linspace(0, self.steps - 1, n_snapshots, dtype=int)

        for integrator_name in self.integrators:
            logging.info(f"Generating snapshots for integrator: {integrator_name}")

            # Create a figure
            fig = plt.figure(figsize=figsize)
            
            # Define GridSpec with 2 columns per snapshot row
            # Adjust 'wspace' and 'hspace' to control spacing between subplots
            # Adjust 'left', 'right', 'top', 'bottom' to control figure margins
            gs = GridSpec(n_fig, 2, figure=fig, 
                        wspace=-0.2,  # Horizontal space between subplots
                        hspace=0.3,  # Vertical space between rows
                        left=0.05, right=0.95, top=0.95, bottom=0.05)  # Margins

            for i, step in enumerate(snapshot_steps):
                # Current simulation time in physical units (Myr)
                current_time = self.times[step] * self.time_scale

                # Plot for x-y plane
                ax_xy = fig.add_subplot(gs[min(i,n_fig-1), 0])
                star_positions_xy = self.positions[integrator_name][step]  # [N, 3]
                ax_xy.scatter(
                    star_positions_xy[:, 0] * self.length_scale,
                    star_positions_xy[:, 1] * self.length_scale,
                    s=1, color='black', alpha=0.5
                )

                # Plot perturbers if present
                if hasattr(self.galaxy, 'perturbers') and len(self.galaxy.perturbers) > 0:
                    if integrator_name in self.perturbers_positions:
                        for pertIndex in range(self.perturbers_positions[integrator_name].shape[0]):
                            perturbers_pos_xy = self.perturbers_positions[integrator_name][pertIndex,step]
                            ax_xy.plot(
                                perturbers_pos_xy[0] * self.length_scale,
                                perturbers_pos_xy[1] * self.length_scale,
                                marker=['*', 'p', 'h', '8', 'D', 'P'][pertIndex%6], markersize=12,
                                color='red', label=f'Perturber {pertIndex+1}'
                            )
                            if i == 0 or independantFig:
                                ax_xy.legend(fontsize=10, loc='upper right')

                ax_xy.set_xlim(-15 * self.length_scale, 15 * self.length_scale)
                ax_xy.set_ylim(-15 * self.length_scale, 15 * self.length_scale)
                ax_xy.set_xlabel('x (kpc)', fontsize=14)
                ax_xy.set_ylabel('y (kpc)', fontsize=14)
                ax_xy.set_title(f'x-y Plane at t = {current_time:.2f} Myr', fontsize=16)
                ax_xy.grid(True, linestyle='--', alpha=0.5)
                ax_xy.set_aspect('equal')

                # Plot for x-z plane
                ax_xz = fig.add_subplot(gs[min(i,n_fig-1), 1])
                star_positions_xz = self.positions[integrator_name][step]  # [N, 3]
                ax_xz.scatter(
                    star_positions_xz[:, 0] * self.length_scale,
                    star_positions_xz[:, 2] * self.length_scale,
                    s=1, color='black', alpha=0.5
                )

                # Plot perturber if present
                if hasattr(self.galaxy, 'perturbers') and len(self.galaxy.perturbers) > 0:
                    if integrator_name in self.perturbers_positions:
                        for pertIndex in range(self.perturbers_positions[integrator_name].shape[0]):
                            perturbers_pos_xz = self.perturbers_positions[integrator_name][pertIndex,step]
                            ax_xz.plot(
                                perturbers_pos_xz[0] * self.length_scale,
                                perturbers_pos_xz[2] * self.length_scale,
                                marker=['*', 'p', 'h', '8', 'D', 'P'][pertIndex%6], markersize=12, color='red', label=f'Perturber {pertIndex+1}'
                            )
                            if i == 0 or independantFig:
                                ax_xz.legend(fontsize=10, loc='upper right')

                ax_xz.set_xlim(-15 * self.length_scale, 15 * self.length_scale)
                ax_xz.set_ylim(-5 * self.length_scale, 5 * self.length_scale)
                ax_xz.set_xlabel('x (kpc)', fontsize=14)
                ax_xz.set_ylabel('z (kpc)', fontsize=14)
                ax_xz.set_title(f'x-z Plane at t = {current_time:.2f} Myr', fontsize=16)
                ax_xz.grid(True, linestyle='--', alpha=0.5)
                ax_xz.set_aspect('equal')

                if independantFig:
                    dir_path = os.path.join(self.results_dir, f'{integrator_name.lower()}_galaxy_{len(self.galaxy.perturbers)}_pert_{n_snapshots}_snapshots')
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)
                    filename = lambda i : f'{'0'*(len(str(n_snapshots)) - len(str(i)))+str(i)}.png'
                    plt.savefig(os.path.join(dir_path, filename(i+1)), bbox_inches='tight')
                    logging.info(f"Snapshot {i+1}/{n_snapshots} saved.")
                    fig.clear()
            
            if not independantFig:
                filename = f'galaxy_{len(self.galaxy.perturbers)}_pert_snapshots_{integrator_name.lower()}.png'
                plt.savefig(os.path.join(self.results_dir, filename), bbox_inches='tight')
                logging.info(f"Galaxy snapshots for {integrator_name} saved to '{os.path.join(self.results_dir,filename)}'.")
            else:
                logging.info(f"Galaxy snapshots for {integrator_name} saved to '{dir_path}'.")
                self.create_animation(dir_path)
            
            plt.close()

    def plot_rotation_curve(self):
        """
        Generate and save the galaxy rotation curve (speed) as a function of R with the velocity dispersion.
        """
        logging.info("Generating galaxy rotation curve plot.")

        for integrator_name in self.integrators:
            logging.info(f"Processing rotation curve for integrator: {integrator_name}")

            # Define radial bins
            R_min = 0.0
            R_max = 15.0  # Adjust based on simulation data
            num_bins = 50
            R_bins = np.linspace(R_min, R_max, num_bins + 1)
            R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])

            # Compute theoretical circular velocity at bin centers
            v_c_theoretical = self.galaxy.circular_velocity(R_centers)

            # From simulation data at the final time step
            positions = self.positions[integrator_name][-1]  # [N, 3]
            velocities = self.velocities[integrator_name][-1]  # [N, 3]

            # Compute cylindrical coordinates
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            R = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)

            # Compute velocities in cylindrical coordinates
            v_x = velocities[:, 0]
            v_y = velocities[:, 1]
            v_z = velocities[:, 2]

            with np.errstate(divide='ignore', invalid='ignore'):
                v_R = (x * v_x + y * v_y) / R
                v_phi = (x * v_y - y * v_x) / R

            # Handle division by zero for R=0
            v_R = np.nan_to_num(v_R)
            v_phi = np.nan_to_num(v_phi)

            # Bin the stars into radial bins
            indices = np.digitize(R, R_bins)

            # Initialize lists to store mean v_phi and dispersion
            mean_v_phi = []
            std_v_phi = []

            for i in range(1, len(R_bins)):
                idx = np.where(indices == i)[0]
                if len(idx) > 0:
                    v_phi_bin = v_phi[idx]
                    mean_v_phi_bin = np.mean(v_phi_bin)
                    std_v_phi_bin = np.std(v_phi_bin)
                    mean_v_phi.append(mean_v_phi_bin)
                    std_v_phi.append(std_v_phi_bin)
                else:
                    mean_v_phi.append(np.nan)
                    std_v_phi.append(np.nan)

            # Convert to arrays
            mean_v_phi = np.array(mean_v_phi)
            std_v_phi = np.array(std_v_phi)

            # Convert units to km/s
            R_centers_kpc = R_centers * self.length_scale  # in kpc
            v_c_theoretical_kms = v_c_theoretical * self.velocity_scale_kms  # in km/s
            mean_v_phi_kms = mean_v_phi * self.velocity_scale_kms  # in km/s
            std_v_phi_kms = std_v_phi * self.velocity_scale_kms  # in km/s

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(R_centers_kpc, v_c_theoretical_kms, label='Theoretical Circular Velocity', color='blue', linewidth=2)
            plt.errorbar(R_centers_kpc, mean_v_phi_kms, yerr=std_v_phi_kms, fmt='o', color='red',
                         label='Mean $v_\\phi$ with Dispersion', capsize=3)
            plt.xlabel('Radius $R$ (kpc)', fontsize=14)
            plt.ylabel('Velocity (km/s)', fontsize=14)
            plt.title(f'Galaxy Rotation Curve ({integrator_name})', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            filename = f'rotation_curve_{integrator_name.lower()}.png'
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()
            logging.info(f"Rotation curve plot for {integrator_name} saved to '{self.results_dir}/{filename}'.")

    def create_animation(self, path):
        if not os.path.exists(path):
            raise NameError(f"Path '{path}' is unknown. Please check the path.")
        
        frames_nb = len([fn for fn in os.listdir(path) if os.path.isfile(os.path.join(path,fn)) and str.endswith(fn, '.png')])
        (
            ffmpeg
            .input(os.path.join(path,'*.png'), pattern_type='glob', framerate=25)
            .output(os.path.join(path,f"Animation_{frames_nb}_snapshots.mp4"))
            .run()
        )
        logging.info(f"Animation_{frames_nb}_snapshots.mp4 ({0.041*frames_nb}s) saved in {path}.")

    def get_energy_difference(self) -> dict:
        """
        Compute the absolute difference in total energy between the final and initial time steps
        for each integrator.

        Returns:
            dict: A dictionary mapping integrator names to their absolute energy difference.
        """
        logging.info("Calculating absolute energy differences |E(t_f) - E(t_i)| for each integrator.")
        energy_diff = {}
        for integrator in self.integrators:
            if integrator in self.total_energy and len(self.total_energy[integrator]) >= 2:
                E_initial = self.total_energy[integrator][0]
                E_final = self.total_energy[integrator][-1]
                diff = abs(E_final - E_initial)
                energy_diff[integrator] = diff
                logging.info(f"Integrator '{integrator}': |E(t_f) - E(t_i)| = {diff:.6e}")
            else:
                logging.warning(f"Total energy data for integrator '{integrator}' is incomplete or missing.")
                energy_diff[integrator] = None
        return energy_diff



def run_single_integrator(integrator_name, galaxy, dt, steps):
    """
    Run a single integrator and return the results.

    Parameters:
        integrator_name (str): Name of the integrator ('Leapfrog', 'RK4', 'Yoshida').
        galaxy (Galaxy): A deep copy of the Galaxy instance.
        dt (float): Time step.
        steps (int): Number of integration steps.

    Returns:
        dict: A dictionary containing the results of the integration.
    """
    integrator = Integrator()
    simulation_results = {}

    if integrator_name == 'Leapfrog':
        pos, vel, energy, Lz, energies_BH, pos_BH, vel_BH, Lz_BH, total_energy, energy_error = integrator.leapfrog(
            galaxy.particles, galaxy, dt, steps
        )
    elif integrator_name == 'RK4':
        pos, vel, energy, Lz, energies_BH, pos_BH, vel_BH, Lz_BH, total_energy, energy_error = integrator.rk4(
            galaxy.particles, galaxy, dt, steps
        )
    elif integrator_name == 'Yoshida':
        pos, vel, energy, Lz, energies_BH, pos_BH, vel_BH, Lz_BH, total_energy, energy_error = integrator.yoshida(
            galaxy.particles, galaxy, dt, steps
        )
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")

    # Store the results in a dictionary
    simulation_results['integrator_name'] = integrator_name
    simulation_results['positions'] = pos
    simulation_results['velocities'] = vel
    simulation_results['energies'] = energy
    simulation_results['angular_momenta'] = Lz
    simulation_results['energies_BH'] = energies_BH
    simulation_results['angular_momenta_BH'] = Lz_BH
    simulation_results['total_energy'] = total_energy
    simulation_results['energy_error'] = energy_error
    simulation_results['positions_BH'] = pos_BH
    simulation_results['velocities_BH'] = vel_BH

    return simulation_results
