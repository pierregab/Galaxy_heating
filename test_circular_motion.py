# test_circular_orbit.py

import logging
import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from galaxy import Galaxy
from particle import Particle
import os
import matplotlib.patches as patches      # For rectangle patches
from matplotlib.patches import ConnectionPatch  # For connecting lines
import matplotlib.ticker as ticker        # For tick formatting
from matplotlib.gridspec import GridSpec    # For advanced subplot layouts

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable LaTeX rendering in Matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
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
    # Initial Conditions for Circular Orbit
    # ============================================================

    # Initialize Galaxy with default parameters
    mass = 1.0        # Mass of the galaxy (normalized)
    a = 2.0           # Radial scale length
    b = 0.1           # Vertical scale length
    epsilon = 0.01    # Softening length for force smoothing

    # Create Galaxy instance
    galaxy = Galaxy(mass=mass, a=a, b=b, epsilon=epsilon)

    # Define initial orbital parameters
    R0 = 10.0          # Initial radius for circular orbit (normalized)
    alpha = 0.0        # No radial displacement for perfect circular orbit
    max_iterations = 100
    use_schwarzschild = False

    # Initialize the single star on a circular orbit
    galaxy.initialize_stars(N=1, Rmax=R0, alpha=alpha, max_iterations=max_iterations, use_schwarzschild=use_schwarzschild)

    # Retrieve R of the star 
    x_star = galaxy.particles[0].position[0]
    y_star = galaxy.particles[0].position[1]
    R_star = np.sqrt(x_star**2 + y_star**2)

    # Extract initial positions, velocities, and masses from Galaxy.particles
    if len(galaxy.particles) != 1:
        logging.error(f"Expected 1 particle after initialization, found {len(galaxy.particles)}.")
        return

    particle = galaxy.particles[0]
    initial_position = particle.position.copy().reshape(1, 3)  # Shape: [1, 3]
    initial_velocity = particle.velocity.copy().reshape(1, 3)  # Shape: [1, 3]
    masses = np.array([particle.mass])                       # Shape: [1]

    # Compute theoretical circular velocity at R0
    # Since the star is on a circular orbit, its velocity should match the circular velocity
    # Extract R from initial_position
    x0, y0, z0 = initial_position[0]
    R_initial = np.sqrt(x0**2 + y0**2)
    v_c = galaxy.circular_velocity(R_initial)

    # Theoretical orbital frequency and period
    omega_orbit = v_c / R_initial
    T_expected = 2 * np.pi / omega_orbit

    logging.info(f"Theoretical circular orbital frequency (omega_orbit): {omega_orbit:.5f} (1/Myr)")
    logging.info(f"Theoretical orbital period (T_expected): {T_expected:.5f} Myr")

    # ============================================================
    # Simulation Parameters
    # ============================================================

    num_periods = 100        # Number of periods to simulate
    dt = 0.01               # Time step (Myr)
    t_max = T_expected * num_periods  # Total simulation time (Myr)

    steps_per_period = int(round(T_expected / dt))  # Number of steps in one period
    total_steps = int(round(t_max / dt))             # Total number of steps

    steps_final_period_start = (num_periods - 1) * steps_per_period  # Starting index for the final period
    steps_final_period_end = num_periods * steps_per_period            # Ending index for the final period

    # Ensure that t_max corresponds to an integer number of steps
    if steps_final_period_end > total_steps:
        logging.warning(f"Final period end index ({steps_final_period_end}) exceeds total steps ({total_steps}). Adjusting to total steps.")
        steps_final_period_end = total_steps
        steps_final_period_start = total_steps - steps_per_period

    # Define the three integrators to compare
    selected_integrators = ['RK4', 'Yoshida', 'Leapfrog']  # Using RK4, Yoshida, and Leapfrog integrators

    # Create Simulation instance
    simulation = Simulation(
        galaxy=galaxy,
        dt=dt,
        t_max=t_max,
        integrators=selected_integrators,
        paralellised=True
    )

    # ============================================================
    # Run the Simulation
    # ============================================================

    logging.info("Running circular orbit simulation...")
    simulation.run()

    # ============================================================
    # Extract and Analyze Simulation Data
    # ============================================================

    # Initialize dictionary to store orbit paths for each integrator
    orbit_paths = {}

    for integrator in selected_integrators:
        # Extract the position data for the integrator
        positions_over_time = simulation.positions.get(integrator, [])

        # Check if positions_over_time is empty
        if isinstance(positions_over_time, list):
            if len(positions_over_time) == 0:
                logging.error(f"No position data found for integrator '{integrator}'.")
                continue
            # For each timestep, extract the (x, y, z) position
            x_positions = np.array([pos[0, 0] for pos in positions_over_time])
            y_positions = np.array([pos[0, 1] for pos in positions_over_time])
            z_positions = np.array([pos[0, 2] for pos in positions_over_time])
        elif isinstance(positions_over_time, np.ndarray):
            if positions_over_time.size == 0:
                logging.error(f"No position data found for integrator '{integrator}'.")
                continue
            # Assuming positions_over_time has shape [num_time_steps, N_particles, 3]
            if positions_over_time.ndim == 3:
                x_positions = positions_over_time[:, 0, 0]
                y_positions = positions_over_time[:, 0, 1]
                z_positions = positions_over_time[:, 0, 2]
            elif positions_over_time.ndim == 2 and positions_over_time.shape[1] == 3:
                x_positions = positions_over_time[:, 0]
                y_positions = positions_over_time[:, 1]
                z_positions = positions_over_time[:, 2]
            else:
                logging.error(f"Unexpected shape for positions_over_time for integrator '{integrator}': {positions_over_time.shape}")
                continue
        else:
            logging.error(f"Unsupported type for positions_over_time for integrator '{integrator}': {type(positions_over_time)}")
            continue

        # Store the full orbit path
        x_full = x_positions
        y_full = y_positions

        orbit_paths[integrator] = (x_full, y_full)

        logging.info(f"Integrator '{integrator}' orbit path extracted with {len(x_full)} points.")

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
    # Ensure 'Figs' Directory Exists
    # ============================================================

    # Ensure the 'Figs' directory exists
    os.makedirs('Figs', exist_ok=True)

    # ============================================================
    # Generate Scientific Plot: Orbit Paths Comparison and Radial Error
    # ============================================================

    # Create a figure with GridSpec for advanced layout
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 2])

    # -------------------------------
    # First Row: Main Orbit Paths and Zoomed Plot
    # -------------------------------
    # Main Orbit Paths Plot (Left)
    ax_main = fig.add_subplot(gs[0, 0])

    # Plot theoretical perfect circular orbit on the main axis
    theta = np.linspace(0, 2 * np.pi, 500)
    x_theoretical = R_star * np.cos(theta)
    y_theoretical = R_star * np.sin(theta)
    ax_main.plot(x_theoretical, y_theoretical, 'k--', label='Theoretical Circle ($R_0$)', linewidth=2)

    # Plot simulated orbits for each integrator on the main axis
    for integrator in selected_integrators:
        if integrator not in orbit_paths:
            continue
        x_final, y_final = orbit_paths[integrator]

        ax_main.plot(
            x_final,
            y_final,
            label=f'{integrator} Integrator',
            color=color_map.get(integrator, None),
            alpha=0.8
        )

    # Set axis limits to focus on the orbit
    ax_main.set_xlim(-1.1 * R_star, 1.1 * R_star)
    ax_main.set_ylim(-1.1 * R_star, 1.1 * R_star)

    # Ensure equal aspect ratio
    ax_main.set_aspect('equal', 'box')

    ax_main.set_title(r'Circular Orbit Simulation: Orbit Paths Comparison')
    ax_main.set_xlabel(r'$x$ (Normalized Units)')
    ax_main.set_ylabel(r'$y$ (Normalized Units)')
    ax_main.legend(loc='upper left', framealpha=1)
    ax_main.grid(True)

    # Add a rectangle on the main plot to indicate the zoomed area
    inset_zoom_range = 0.1  # Adjusted for better visibility
    center_x, center_y = 0, 0  # Center the zoom around (0, 0) or another point as desired


    # Zoomed-In Orbit Plot (Right)
    ax_zoom = fig.add_subplot(gs[0, 1])

    # Define the zoom window around the theoretical radius R0
    inset_zoom_range = 0.000001  # Adjusted for better visibility
    center_x, center_y = R_star, 0  # Center the zoom around (R_star, 0)

    # Plotting the zoomed-in section on the right axis
    for integrator in selected_integrators:
        if integrator not in orbit_paths:
            continue
        x_final, y_final = orbit_paths[integrator]
        ax_zoom.plot(
            x_final,
            y_final,
            label=f'{integrator} Integrator',
            color=color_map.get(integrator, None),
            alpha=0.5
        )

    # Plot the theoretical circle on the zoomed axis
    ax_zoom.plot(x_theoretical, y_theoretical, 'k--', linewidth=3)

    # Set tighter axis limits for the zoomed plot
    ax_zoom.set_xlim(center_x - inset_zoom_range, center_x + inset_zoom_range)
    ax_zoom.set_ylim(center_y - inset_zoom_range, center_y + inset_zoom_range)


    ax_zoom.set_title('Zoomed Orbit Paths')
    ax_zoom.set_xlabel(r'$x$ (Normalized Units)')
    ax_zoom.set_ylabel(r'$y$ (Normalized Units)')
    ax_zoom.legend(loc='upper left', framealpha=1)
    ax_zoom.grid(True)

    # Format tick labels (optional)
    ax_zoom.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.8f'))
    ax_zoom.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.8f'))

    # Rotate x tick labels for better readability
    for tick in ax_zoom.get_xticklabels():
        tick.set_rotation(45)

    # Adjust tick parameters for better visibility
    ax_zoom.tick_params(axis='both', which='major', labelsize=12)

    # Add connecting lines between the rectangle on the main plot and the zoomed plot
    # Get the positions of the rectangle corners
    rect_corners = [
        (center_x - inset_zoom_range, center_y - inset_zoom_range),  # Lower-left
        (center_x + inset_zoom_range, center_y - inset_zoom_range),  # Lower-right
        (center_x + inset_zoom_range, center_y + inset_zoom_range),  # Upper-right
        (center_x - inset_zoom_range, center_y + inset_zoom_range)   # Upper-left
    ]

    # Define connection points from main plot to zoomed plot
    # We'll connect the upper-right and lower-right corners to the zoomed plot
    connections = [
        (rect_corners[1], (center_x - inset_zoom_range, center_y - inset_zoom_range)),  # Lower-right to lower-left of zoom
        (rect_corners[2], (center_x - inset_zoom_range, center_y + inset_zoom_range))   # Upper-right to upper-left of zoom
    ]

    for start, end in connections:
        # Create a ConnectionPatch
        con = ConnectionPatch(
            xyA=start, coordsA=ax_main.transData,
            xyB=end, coordsB=ax_zoom.transData,
            color='red', linewidth=1.5, linestyle='--'
        )
        fig.add_artist(con)

    # -------------------------------
    # Second Row: Radial Error Plot (Log Scale)
    # -------------------------------
    ax_error = fig.add_subplot(gs[1, :])  # Span both columns

    # Calculate the initial phase
    phi = np.arctan2(y0, x0)

    # Function to compute theoretical positions based on simulation time steps
    def compute_theoretical_positions(x0, y0, omega, times, phi):
        theta = omega * times + phi
        x_theory = R_star * np.cos(theta)
        y_theory = R_star * np.sin(theta)
        return x_theory, y_theory

    # Generate time array for the entire simulation
    times_full = np.linspace(0, t_max, total_steps, endpoint=False)  # t_max excluded to match steps

    # Compute theoretical positions for the entire simulation with initial phase
    x_theory_full, y_theory_full = compute_theoretical_positions(x0, y0, omega_orbit, times_full, phi)

    # Initialize dictionary to store radial errors
    radial_errors = {}

    for integrator in selected_integrators:
        if integrator not in orbit_paths:
            continue
        x_final, y_final = orbit_paths[integrator]

        # Check if the lengths match
        len_sim = len(x_final)
        len_theory = len(x_theory_full)
        if len_sim != len_theory:
            min_len = min(len_sim, len_theory)
            logging.warning(f"Integrator '{integrator}': Simulation data length ({len_sim}) != Theoretical data length ({len_theory}). Trimming to min length ({min_len}).")
            x_final_trimmed = x_final[:min_len]
            y_final_trimmed = y_final[:min_len]
            x_theory_trimmed = x_theory_full[:min_len]
            y_theory_trimmed = y_theory_full[:min_len]
        else:
            x_final_trimmed = x_final
            y_final_trimmed = y_final
            x_theory_trimmed = x_theory_full
            y_theory_trimmed = y_theory_full

        # Compute radial error as |r_sim - R_star|
        r_sim = np.sqrt(x_final_trimmed**2 + y_final_trimmed**2)
        error_r = np.abs(r_sim - R_star)
        radial_errors[integrator] = error_r

    # Compute the number of revolutions for the entire simulation
    revolutions_full = times_full / T_expected  # e.g., t = T_expected corresponds to 1 revolution

    # Plot radial errors on the bottom subplot using a logarithmic y-axis
    for integrator in selected_integrators:
        if integrator not in radial_errors:
            continue
        error_r = radial_errors[integrator]
        revolutions_trimmed = revolutions_full[:len(error_r)]  # Align revolutions array with error array
        ax_error.plot(
            revolutions_trimmed,
            error_r,
            label=f'{integrator} Integrator',
            linestyle='-',
            color=color_map.get(integrator, None),
            alpha=0.8
        )

    ax_error.set_yscale('log')
    ax_error.set_xlabel('Revolution')
    ax_error.set_ylabel('Radial Error |$r_{sim}$ - $R_0$| (Log Scale)')
    ax_error.set_title('Radial Error Comparison Over All Revolutions (Log Scale)')
    ax_error.legend(loc='upper right', framealpha=1)
    ax_error.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Enhance tick labels for better readability
    ax_error.tick_params(axis='both', which='major', labelsize=12)
    ax_error.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_error.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))

    # ============================================================
    # Save the orbit paths comparison plot with zoomed plot on the right and radial error below
    # ============================================================

    plt.savefig('Figs/circular_orbit_comparison.png', dpi=300)
    plt.close()

    # ============================================================
    # Log Summary Results
    # ============================================================

    logging.info("Orbit paths comparison plot with radial errors saved as 'Figs/circular_orbit_comparison.png'.")

if __name__ == '__main__':
    main()
