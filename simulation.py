from system import System
from galaxy import Galaxy
from integrators import Integrator
import os  # Import the os module for directory operations
import timeit
import ffmpeg
import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set up professional logging
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

    def __init__(self, galaxy:Galaxy, dt:float=0.05, t_max:float=250.0, integrators:list[str]=['Leapfrog', 'RK4']) -> None:
        """
        Initialize the Simulation.

        Parameters:
            galaxy (Galaxy): The galaxy instance.
            dt (float): Time step (dimensionless).
            t_max (float): Total simulation time (dimensionless).
            integrators (list): List of integrators to run. Options: 'Leapfrog', 'RK4'.
        """
        super().__init__(self.__class__.__name__)
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

        # Validate integrators
        valid_integrators = ['Leapfrog', 'RK4']
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
        Run the simulation using the selected integrators.
        """
        logging.info("Starting the simulation.")

        for integrator_name in self.integrators:
            # Reset the system before each integrator run
            self.reset_system()

            if integrator_name == 'Leapfrog':
                # Leapfrog Integration Timing
                def run_leapfrog():
                    return self.integrator.leapfrog(self.galaxy.particles, self.galaxy, self.dt, self.steps)

                start_time = timeit.default_timer()
                pos, vel, energy, Lz, energies_BH, pos_BH, vel_BH, Lz_BH = run_leapfrog()
                total_time = timeit.default_timer() - start_time
                average_time = total_time / self.steps
                logging.info(f"Leapfrog integration took {total_time:.3f} seconds in total.")
                logging.info(f"Average time per step (Leapfrog): {average_time*1e3:.6f} ms.")
                self.execution_times['Leapfrog'] = average_time * 1e3  # in ms

                # Store results
                self.positions['Leapfrog'] = pos
                self.velocities['Leapfrog'] = vel
                self.energies['Leapfrog'] = energy
                self.angular_momenta['Leapfrog'] = Lz
                self.energies_BH['Leapfrog'] = energies_BH
                self.angular_momenta_BH['Leapfrog'] = Lz_BH
                if pos_BH is not None:
                    self.perturbers_positions['Leapfrog'] = pos_BH
                    self.perturbers_velocities['Leapfrog'] = vel_BH

            elif integrator_name == 'RK4':
                # RK4 Integration Timing
                def run_rk4():
                    return self.integrator.rk4(self.galaxy.particles, self.galaxy, self.dt, self.steps)

                start_time = timeit.default_timer()
                pos, vel, energy, Lz, energies_BH, pos_BH, vel_BH = run_rk4()
                total_time = timeit.default_timer() - start_time
                average_time = total_time / self.steps
                logging.info(f"RK4 integration took {total_time:.3f} seconds in total.")
                logging.info(f"Average time per step (RK4): {average_time*1e3:.6f} ms.")
                self.execution_times['RK4'] = average_time * 1e3  # in ms

                # Store results
                self.positions['RK4'] = pos
                self.velocities['RK4'] = vel
                self.energies['RK4'] = energy
                self.angular_momenta['RK4'] = Lz
                self.energies_BH['RK4'] = energies_BH
                if pos_BH is not None:
                    self.perturbers_positions['RK4'] = pos_BH
                    self.perturbers_velocities['RK4'] = vel_BH

        logging.info("Simulation completed.")

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
            # Ensure that energy data for the integrator exists
            if integrator_name not in self.energies or (hasattr(self.galaxy, 'perturbers') and integrator_name not in self.energies_BH):
                logging.warning(f"Energy data for integrator '{integrator_name}' is incomplete. Skipping.")
                continue

            # Time array in physical units
            times_physical = self.times * self.time_scale  # Time in Myr

            # Stars' energies: [steps, N]
            E_stars = self.energies[integrator_name]  # [steps, N]

            # Sum stars' energies at each step to get total stars' energy
            total_E_stars = np.sum(E_stars, axis=1)  # [steps]

            if hasattr(self.galaxy, 'perturbers') and len(self.galaxy.perturbers):
                # Perturbers' energies: [P, steps]
                E_BH = self.energies_BH[integrator_name]  # [P, steps]

                # System's total energy at each step
                total_E_system = total_E_stars + np.add.reduce(E_BH, axis=0)  # [steps]
            else:
                # If no perturbers, system's total energy is stars' total energy
                total_E_system = total_E_stars  # [steps]

            # Initial total energy
            E_total_initial = total_E_system[0]  # Scalar

            # Compute relative energy error
            relative_E_error = (total_E_system - E_total_initial) / np.abs(E_total_initial)  # [steps]

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
        plt.grid(True)
        plt.yscale('log')  # Using logarithmic scale to capture small errors
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'total_energy_error.png'))
        plt.close()
        logging.info(f"Total energy conservation error plot saved to '{self.results_dir}/total_energy_error.png'.")

    def plot_angular_momentum_error(self) -> None:
        """
        Plot the angular momentum conservation error over time.
        """
        logging.info("Generating angular momentum conservation error plot.")

        plt.figure(figsize=(12, 8))
        for integrator_name in self.integrators:
            # Time array in physical units
            times_physical = self.times * self.time_scale  # Time in Myr

            if len(self.galaxy.perturbers):
                print(self.angular_momenta_BH[integrator_name])
                # Compute angular momentum errors
                L0 = np.concatenate((self.angular_momenta[integrator_name][0], self.angular_momenta_BH[integrator_name][0]))  # [N+P]
                L_error = (np.concatenate((self.angular_momenta[integrator_name],self.angular_momenta_BH[integrator_name]), axis=1) - L0) / np.abs(L0)  # [steps, N+P]
            else:
                # Compute angular momentum errors
                L0 = self.angular_momenta[integrator_name][0] # [N+P]
                L_error = (self.angular_momenta[integrator_name] - L0) / np.abs(L0)  # [steps, N+P]

            # Compute average angular momentum error across all stars
            avg_L_error = np.mean(np.abs(L_error), axis=1)  # [steps]

            plt.plot(times_physical, avg_L_error, label=integrator_name, linewidth=1)

        plt.xlabel('Time (Myr)', fontsize=14)
        plt.ylabel('Relative Angular Momentum Error', fontsize=14)
        plt.title('Angular Momentum Conservation Error Over Time', fontsize=16)
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

        # Compute initial sigma_R_init and sigma_z_init once
        R_c_bins = np.linspace(np.min(self.galaxy.R_c), np.max(self.galaxy.R_c), 10)
        indices = np.digitize(self.galaxy.R_c, R_c_bins)
        R_c_centers = 0.5 * (R_c_bins[:-1] + R_c_bins[1:])

        sigma_R_init = []
        sigma_z_init = []
        for i in range(1, len(R_c_bins)):
            idx = np.where(indices == i)[0]
            if len(idx) > 1:
                # Initial sigma values (from theoretical expressions)
                sigma_R_initial_val = np.mean(self.galaxy.initial_sigma_R[idx])
                sigma_z_initial_val = np.mean(self.galaxy.initial_sigma_z[idx])
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
                x = pos[:, 0]
                y = pos[:, 1]
                z = pos[:, 2]
                R = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)

                # Compute velocities in cylindrical coordinates
                v_x = vel[:, 0]
                v_y = vel[:, 1]
                v_z = vel[:, 2]

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
                        dispersions_R[moment_label],
                        yerr=uncertainties_R[moment_label],
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
                        dispersions_z[moment_label],
                        yerr=uncertainties_z[moment_label],
                        marker=moment_styles[moment_label]['marker_z'],
                        linestyle=moment_styles[moment_label]['linestyle_z'],
                        label=f"{moment_label} σ_z",
                        color=moment_styles[moment_label]['color'],
                        capsize=3,
                        markersize=6,
                        linewidth=1.5
                    )

            # Plot initial theoretical dispersions as solid lines
            plt.plot(R_c_centers, sigma_R_init, 'k-', label='Initial σ_R (Theoretical)', linewidth=2)
            plt.plot(R_c_centers, sigma_z_init, 'k--', label='Initial σ_z (Theoretical)', linewidth=2)

            plt.xlabel('Reference Radius $R_c$ (dimensionless)', fontsize=16)
            plt.ylabel('Velocity Dispersion σ (dimensionless)', fontsize=16)
            plt.title(f'Velocity Dispersions at Different Moments ({integrator_name})', fontsize=18)

            # Place the legend outside the plot to avoid overlapping with data
            plt.legend(fontsize=12, bbox_to_anchor=(1, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect to make space for the legend

            filename = f'velocity_dispersions_{integrator_name.lower()}_moments.png'
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
                    logging.info(f"{integrator_name} - Moment: {moment_label}, R_c = {R_c_centers[i]:.2f}: "
                                f"σ_R = {sigma_R:.4f} ± {sigma_R_unc:.4f}, "
                                f"σ_z = {sigma_z:.4f} ± {sigma_z_unc:.4f}, "
                                f"Initial σ_R = {sigma_R_init_val:.4f}, Initial σ_z = {sigma_z_init_val:.4f}")

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
