# test_spring_motion.py

import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from simulation import Simulation
from galaxy import Galaxy

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable LaTeX rendering in Matplotlib
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

# Change dpi setting for higher resolution plots
plt.rcParams['figure.dpi'] = 150

def main() -> None:
    # ============================================================
    # Initial Conditions for Spring (Vertical Oscillatory) Motion
    # ============================================================

    # Initialize Galaxy with default parameters
    mass = 1.0        # Mass of the galaxy (normalized)
    a = 2.0           # Radial scale length
    b = 0.1           # Vertical scale length (set explicitly)
    epsilon = 0.01    # Softening length for force smoothing

    # Create Galaxy instance
    galaxy = Galaxy(mass=mass, a=a, b=b, epsilon=epsilon)

    # Compute theoretical vertical oscillation frequency and period
    # Using the vertical component of the acceleration near z=0
    # For small z, the motion can be approximated as harmonic: a_z = -omega_z^2 * z
    # Thus, omega_z = sqrt(d2Phi/dz2 at z=0)
    # From the Galaxy class, acceleration is computed; we'll approximate omega_z numerically

    # Small displacement to compute d2Phi/dz2
    delta_z = 1e-5
    pos_plus = np.array([[0.0, 0.0, delta_z]])
    pos_minus = np.array([[0.0, 0.0, -delta_z]])

    # Compute accelerations at z = +delta_z and z = -delta_z
    acc_plus = galaxy.acceleration(pos_plus)    # Shape: [1, 3]
    acc_minus = galaxy.acceleration(pos_minus)  # Shape: [1, 3]

    # Compute second derivative numerically: d2Phi/dz2 ≈ (a_z(z + delta_z) - a_z(z - delta_z)) / (2 * delta_z)
    d2Phi_dz2 = (acc_plus[0, 2] - acc_minus[0, 2]) / (2 * delta_z)

    # Compute omega_z
    try:
        omega_z = np.sqrt(-d2Phi_dz2)  # Ensure the argument is positive
    except ValueError as e:
        logging.error(f"Error computing omega_z: {e}")
        return

    # Theoretical period
    T_expected = 2 * np.pi / omega_z

    logging.info(f"Theoretical vertical oscillation frequency (omega_z): {omega_z:.5f}")
    logging.info(f"Theoretical period (T_expected): {T_expected:.5f} Myr")

    # Initial position and velocity for the single star
    z0 = 0.1  # Initial displacement along z-axis
    initial_position = np.array([[0.0, 0.0, z0]])  # Shape: [1, 3]
    initial_velocity = np.array([[0.0, 0.0, 0.0]])  # Shape: [1, 3]
    masses = np.array([mass])                       # Shape: [1]

    # Initialize the single star with specified conditions using initialize_stars_specific
    galaxy.initialize_stars_specific(positions=initial_position, velocities=initial_velocity, masses=masses)

    # ============================================================
    # Simulation Parameters
    # ============================================================

    num_periods = 1000        # Number of periods to simulate for multiple return events
    dt = 0.01                   # Time step (Myr)
    t_max = T_expected * num_periods  # Total simulation time (Myr)

    # Define the three integrators to compare
    selected_integrators = ['RK4', 'Yoshida', 'Leapfrog']  # Using RK4, Yoshida, and Leapfrog integrators

    # Create Simulation instance
    simulation = Simulation(galaxy=galaxy, dt=dt, t_max=t_max, integrators=selected_integrators, paralellised=True)

    # ============================================================
    # Run the Simulation
    # ============================================================

    logging.info("Running spring motion simulation...")
    simulation.run()

    # ============================================================
    # Extract and Analyze Simulation Data
    # ============================================================

    # Initialize dictionaries to store return times and differences for each integrator
    return_times_dict = {}
    return_differences_dict = {}

    for integrator in selected_integrators:
        # Extract the position data for the integrator
        positions_over_time = simulation.positions.get(integrator, [])

        # Check if positions_over_time is empty
        if isinstance(positions_over_time, list):
            if len(positions_over_time) == 0:
                logging.error(f"No position data found for integrator '{integrator}'.")
                continue
            # Extract z positions over time
            z_positions = np.array([pos[0, 2] for pos in positions_over_time])  # Shape: [num_time_steps]
        elif isinstance(positions_over_time, np.ndarray):
            if positions_over_time.size == 0:
                logging.error(f"No position data found for integrator '{integrator}'.")
                continue
            # Assuming positions_over_time has shape [num_time_steps, N_particles, 3]
            if positions_over_time.ndim == 3:
                z_positions = positions_over_time[:, 0, 2]  # Shape: [num_time_steps]
            elif positions_over_time.ndim == 2 and positions_over_time.shape[1] == 3:
                z_positions = positions_over_time[:, 2]     # Shape: [num_time_steps]
            else:
                logging.error(f"Unexpected shape for positions_over_time for integrator '{integrator}': {positions_over_time.shape}")
                continue
        else:
            logging.error(f"Unsupported type for positions_over_time for integrator '{integrator}': {type(positions_over_time)}")
            continue

        # Extract times
        times = simulation.times  # Assuming simulation.times is a 1D array of shape [num_time_steps]

        # Validate that times and z_positions have the same length
        if len(times) != len(z_positions):
            logging.error(f"Mismatch between times length ({len(times)}) and z_positions length ({len(z_positions)}) for integrator '{integrator}'.")
            logging.error(f"Shape of times: {times.shape}, Shape of z_positions: {z_positions.shape}")
            continue

        # Compute the difference between simulated z and initial z0
        position_difference = z_positions - z0  # Shape: [num_time_steps]

        # Find all peaks in z_positions (assuming each peak corresponds to returning to z0)
        peaks, _ = find_peaks(z_positions)

        if len(peaks) == 0:
            logging.warning(f"No peaks detected in the z-position data for integrator '{integrator}'. The star did not return to the starting position within the simulation time.")
            continue

        # Collect differences at each peak
        return_times = times[peaks]
        return_differences = z_positions[peaks] - z0

        # Store in dictionaries
        return_times_dict[integrator] = return_times
        return_differences_dict[integrator] = return_differences

        # Log each return event (only log the last few returns to avoid excessive logging)
        num_returns_to_log = 10
        if len(return_times) > num_returns_to_log:
            logging.info(f"Integrator '{integrator}' - Last {num_returns_to_log} Returns:")
            for i, (t_return, diff) in enumerate(zip(return_times[-num_returns_to_log:], return_differences[-num_returns_to_log:]), 1):
                logging.info(f"  Return {i}: Time = {t_return:.5f} Myr, Difference = {diff:.5e}")
        else:
            logging.info(f"Integrator '{integrator}' - All Returns:")
            for i, (t_return, diff) in enumerate(zip(return_times, return_differences), 1):
                logging.info(f"  Return {i}: Time = {t_return:.5f} Myr, Difference = {diff:.5e}")

    # ============================================================
    # Define Color Mapping for Integrators
    # ============================================================

    # Define a color for each integrator to ensure consistency across plots
    color_map = {
        'RK4': '#1f77b4',       # Matplotlib's default blue
        'Yoshida': '#ff7f0e',   # Matplotlib's default orange
        'Leapfrog': '#2ca02c'   # Matplotlib's default green
    }

    # ============================================================
    # Generate Scientific Plots
    # ============================================================

    # Define the number of recent returns to display in the z-position plot
    recent_returns_to_display = 5  # Adjust as needed

    # Plot 1: z-position over time with peaks highlighted for each integrator
    plt.figure(figsize=(14, 8))

    for integrator in selected_integrators:
        # Extract z_positions and peaks
        positions_over_time = simulation.positions.get(integrator, [])

        if isinstance(positions_over_time, list):
            if len(positions_over_time) == 0:
                continue
            z_positions = np.array([pos[0, 2] for pos in positions_over_time])
        elif isinstance(positions_over_time, np.ndarray):
            if positions_over_time.size == 0:
                continue
            if positions_over_time.ndim == 3:
                z_positions = positions_over_time[:, 0, 2]
            elif positions_over_time.ndim == 2 and positions_over_time.shape[1] == 3:
                z_positions = positions_over_time[:, 2]
            else:
                continue
        else:
            continue

        times = simulation.times

        # Find all peaks
        peaks, _ = find_peaks(z_positions)

        if len(peaks) == 0:
            logging.warning(f"Integrator '{integrator}' has no peaks to plot.")
            continue

        # Collect differences at each peak
        return_times = times[peaks]
        return_differences = z_positions[peaks] - z0

        # Determine peaks to display
        if len(peaks) > recent_returns_to_display:
            # Select the last few peaks
            display_peaks = peaks[-recent_returns_to_display:]
            # Define buffer time around the first of the recent peaks
            buffer_time = T_expected  # 1 period as buffer
            start_time = times[display_peaks[0]] - buffer_time
            end_time = times[display_peaks[-1]] + buffer_time

            # Create a mask for times within the buffer window
            mask = (times >= start_time) & (times <= end_time)
            subset_times = times[mask]
            subset_z = z_positions[mask]

            # Find peaks within the subset
            subset_peaks, _ = find_peaks(subset_z)

            if len(subset_peaks) == 0:
                logging.warning(f"No peaks found in the subset window for integrator '{integrator}'.")
                continue

            # Use subset_peaks directly without adding any offset
            display_peaks = subset_peaks

            # Update times and z_positions for plotting
            plot_times = subset_times
            plot_z = subset_z
        else:
            display_peaks = peaks
            plot_times = times
            plot_z = z_positions

        # Plot z(t) with the integrator's assigned color
        plt.plot(plot_times, plot_z, label=f'{integrator} Integrator', color=color_map.get(integrator, None))

        # Plot peaks with the same color and a distinct marker
        if len(display_peaks) > 0:
            plt.plot(plot_times[display_peaks], plot_z[display_peaks], "x", label=f'{integrator} Returns', color=color_map.get(integrator, None))

    plt.axhline(y=z0, color='black', linestyle='--', label=r'Initial Position ($z_0$)')
    plt.title(r'Spring Motion Simulation: $z$-Position vs. Time for Different Integrators')
    plt.xlabel(r'Time (Myr)')
    plt.ylabel(r'Position ($z$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the z-position over time plot
    plt.savefig('Figs/spring_motion_z_vs_time_all_integrators.png')
    plt.close()

    # Plot 2: Absolute position difference at each return event for all integrators (log y-axis)
    plt.figure(figsize=(14, 8))

    for integrator in selected_integrators:
        if integrator not in return_differences_dict:
            continue
        return_diffs = return_differences_dict[integrator]
        return_nums = np.arange(1, len(return_diffs) + 1)

        # Plot absolute differences with the integrator's assigned color
        plt.scatter(return_nums, np.abs(return_diffs), label=f'{integrator} Integrator', alpha=0.7, s=10, color=color_map.get(integrator, None))
        plt.plot(return_nums, np.abs(return_diffs), linestyle='-', alpha=0.7, color=color_map.get(integrator, None))

    plt.yscale('log')

    plt.title(r'Spring Motion Simulation: Absolute Position Difference at Returns for Different Integrators')
    plt.xlabel(r'Return Number')
    plt.ylabel(r'Absolute Position Difference $|z_{\mathrm{simulated}} - z_0|$')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    # Save the position difference at returns plot
    plt.savefig('Figs/spring_motion_position_difference_at_returns_log.png')
    plt.close()

    # ============================================================
    # Log Summary Results
    # ============================================================

    for integrator in selected_integrators:
        if integrator not in return_differences_dict:
            continue
        return_diffs = return_differences_dict[integrator]
        max_diff = np.abs(return_diffs).max()
        final_diff = return_diffs[-1]
        logging.info(f"Integrator '{integrator}':")
        logging.info(f"  Number of returns detected: {len(return_diffs)}")
        logging.info(f"  Final return difference: {final_diff:.5e}")
        logging.info(f"  Maximum absolute difference across returns: {max_diff:.2e}")

if __name__ == '__main__':
    main()
