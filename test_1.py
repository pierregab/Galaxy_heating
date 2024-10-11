import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Constants
G = 1.0        # Gravitational constant
M = 1.0       # Mass of the galaxy
a = 2.5        # Radial scale length
b = 1/20 * a        # Vertical scale length

# Time parameters
dt = 0.001     # Time step
t_max = 500.0  # Total simulation time
steps = int(t_max / dt)

# Functions for the potential and acceleration
def potential(R, z):
    denom = np.sqrt(R**2 + (a + np.sqrt(z**2 + b**2))**2)
    return -G * M / denom

def acceleration(pos):
    x, y, z = pos
    R = np.sqrt(x**2 + y**2)
    z_term = np.sqrt(z**2 + b**2)
    denom = (R**2 + (a + z_term)**2)**1.5
    ax = -G * M * x / denom
    ay = -G * M * y / denom
    az = -G * M * (a + z_term) * z / (z_term * denom)
    return np.array([ax, ay, az])

# Theoretical Calculations for 1D Motion

# Initial conditions for 1D motion
r0 = 5.0   # Initial radial position
pos_initial = np.array([r0, 0.0, 0.0])  # Starting at x = r0, y = 0, z = 0
vel_initial = np.array([0.0, 0.0, 0.0])  # No initial velocity

# Calculate angular frequency (omega) using harmonic approximation
# Compute k = dF/dx at x = r0
k = (G * M * (2 * r0**2 - (a + b)**2)) / ((r0**2 + (a + b)**2)**(5/2))
omega = np.sqrt(k)  # Assuming unit mass

# Theoretical period
T_theoretical = 2 * np.pi / omega

# Arrays to store data
positions = np.zeros((steps, 3))
velocities = np.zeros((steps, 3))
energies = np.zeros(steps)
angular_momenta = np.zeros(steps)
times = np.linspace(0, t_max, steps)

# Leapfrog integrator for circular orbit
# Initialize for circular orbit
pos = np.array([r0, 0.0, 0.0])  # Starting at x = r0, y = 0, z = 0
R = np.sqrt(pos[0]**2 + pos[1]**2)
z = pos[2]
# Calculate the circular velocity needed
v_circular = np.sqrt(G * M * R**2 / (R**2 + (a + np.sqrt(z**2 + b**2))**2)**1.5)
vel = np.array([0.0, v_circular, 0.0])  # Velocity in y-direction

# Initialize storage for circular orbit
positions_circ = np.zeros((steps, 3))
velocities_circ = np.zeros((steps, 3))
energies_circ = np.zeros(steps)
angular_momenta_circ = np.zeros(steps)

# Leapfrog integrator for circular orbit
vel_half = vel + 0.5 * dt * acceleration(pos)

for i in range(steps):
    # Update position
    pos += dt * vel_half
    positions_circ[i] = pos
    # Update acceleration
    acc = acceleration(pos)
    # Update velocity
    vel_half += dt * acc
    velocities_circ[i] = vel_half - 0.5 * dt * acc  # Average velocity
    # Compute energy and angular momentum
    R = np.sqrt(pos[0]**2 + pos[1]**2)
    z = pos[2]
    v = np.linalg.norm(velocities_circ[i])
    energies_circ[i] = 0.5 * v**2 + potential(R, z)
    angular_momenta_circ[i] = pos[0]*velocities_circ[i][1] - pos[1]*velocities_circ[i][0]  # Lz component

# 5. Simulate 1D Movement with No Radial Velocity
# Initialize for 1D motion
pos_1d = pos_initial.copy()
vel_1d = vel_initial.copy()

# Reset arrays for 1D motion
positions_1d = np.zeros((steps, 3))
velocities_1d = np.zeros((steps, 3))
energies_1d = np.zeros(steps)

# Leapfrog integrator for 1D motion
vel_half_1d = vel_1d + 0.5 * dt * acceleration(pos_1d)

for i in range(steps):
    # Update position
    pos_1d += dt * vel_half_1d
    positions_1d[i] = pos_1d
    # Update acceleration
    acc = acceleration(pos_1d)
    # Update velocity
    vel_half_1d += dt * acc
    velocities_1d[i] = vel_half_1d - 0.5 * dt * acc
    # Compute energy
    R = np.sqrt(pos_1d[0]**2 + pos_1d[1]**2)
    z = pos_1d[2]
    v = np.linalg.norm(velocities_1d[i])
    energies_1d[i] = 0.5 * v**2 + potential(R, z)

# Analyze position deviation over time
# Find extrema positions
positions_x = positions_1d[:,0]

# Use scipy.signal.argrelextrema to find local maxima and minima
max_indices = argrelextrema(positions_x, np.greater)[0]
min_indices = argrelextrema(positions_x, np.less)[0]

# Extract times and positions at maxima and minima
times_max = times[max_indices]
times_min = times[min_indices]
positions_max = positions_x[max_indices]
positions_min = positions_x[min_indices]

# Combine maxima and minima for plotting
orbit_numbers_max = np.arange(1, len(positions_max) + 1)
orbit_numbers_min = np.arange(1, len(positions_min) + 1)

# Calculate theoretical maxima and minima (Assuming harmonic oscillation)
positions_theoretical_max = r0  # + amplitude (harmonic)
positions_theoretical_min = -r0  # - amplitude (harmonic)

# Compute percentage differences
# Avoid division by zero by ensuring theoretical positions are not zero
percentage_diff_max = ((positions_max - positions_theoretical_max) / positions_theoretical_max) * 100
percentage_diff_min = ((positions_min - positions_theoretical_min) / positions_theoretical_min) * 100

# --- Plotting ---

# --- Summary Graph 1: Circular Orbit ---

# change dpi 
plt.rcParams['figure.dpi'] = 150

# Calculate normalized energy and angular momentum errors
initial_energy_circ = energies_circ[0]
normalized_energy_error_circ = (energies_circ - initial_energy_circ) / np.abs(initial_energy_circ) * 100  # Percentage error

initial_Lz_circ = angular_momenta_circ[0]
normalized_Lz_error_circ = (angular_momenta_circ - initial_Lz_circ) / np.abs(initial_Lz_circ) * 100  # Percentage error

# To avoid taking log of zero or negative values, add a small epsilon
epsilon = 1e-20
normalized_energy_error_circ_safe = np.abs(normalized_energy_error_circ) + epsilon
normalized_Lz_error_circ_safe = np.abs(normalized_Lz_error_circ) + epsilon

# Create the Circular Orbit Summary Figure
fig_circ, axs_circ = plt.subplots(2, 1, figsize=(12, 12))

# 1. Orbit Trajectory
axs_circ[0].plot(positions_circ[:,0], positions_circ[:,1], color='blue', lw=1, label='Orbit Path')
axs_circ[0].plot(0, 0, 'ro', label='Galactic Center')
axs_circ[0].set_xlabel('x')
axs_circ[0].set_ylabel('y')
axs_circ[0].set_title('Circular Orbit Trajectory')
axs_circ[0].axis('equal')
axs_circ[0].legend()
axs_circ[0].grid(True, linestyle='--', alpha=0.7)

# 2. Normalized Energy and Angular Momentum Errors Over Time (Log Scale)
ax2 = axs_circ[1]  # Primary y-axis
color1 = 'green'
ax2.set_xlabel('Time')
ax2.set_ylabel('Normalized Energy Error (%)', color=color1)
ax2.set_yscale('log')  # Set y-axis to logarithmic scale
ax2.plot(times, normalized_energy_error_circ_safe, color=color1, lw=1, label='Energy Error')
ax2.tick_params(axis='y', labelcolor=color1)
ax2.grid(True, linestyle='--', alpha=0.7)

ax3 = ax2.twinx()  # Secondary y-axis
color2 = 'purple'
ax3.set_ylabel('Normalized Angular Momentum Error (%)', color=color2)
ax3.set_yscale('log')  # Set y-axis to logarithmic scale
ax3.plot(times, normalized_Lz_error_circ_safe, color=color2, lw=1, label='Angular Momentum Error')
ax3.tick_params(axis='y', labelcolor=color2)

# Add legends
lines_1, labels_1 = ax2.get_legend_handles_labels()
lines_2, labels_2 = ax3.get_legend_handles_labels()
axs_circ[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

axs_circ[1].set_title('Normalized Energy and Angular Momentum Errors Over Time (Circular Orbit)')

fig_circ.tight_layout()
plt.show()

# --- Summary Graph 2: 1D Orbit ---

# Calculate normalized energy error for 1D motion
initial_energy_1d = energies_1d[0]
normalized_energy_error_1d = (energies_1d - initial_energy_1d) / np.abs(initial_energy_1d) * 100  # Percentage error

# Calculate absolute percentage error of extrema positions
absolute_percentage_diff_max = np.abs(percentage_diff_max)
absolute_percentage_diff_min = np.abs(percentage_diff_min)

# Combine maxima and minima absolute percentage differences
orbit_numbers_max = np.arange(1, len(absolute_percentage_diff_max) + 1)
orbit_numbers_min = np.arange(1, len(absolute_percentage_diff_min) + 1)

# To avoid log of zero, add epsilon
normalized_energy_error_1d_safe = np.abs(normalized_energy_error_1d) + epsilon
absolute_percentage_diff_max_safe = absolute_percentage_diff_max + epsilon
absolute_percentage_diff_min_safe = absolute_percentage_diff_min + epsilon

# Create the 1D Orbit Summary Figure
fig_1d, axs_1d = plt.subplots(3, 1, figsize=(12, 18))

# 1. Phase Space Trajectory
axs_1d[0].plot(positions_1d[:,0], velocities_1d[:,0], color='teal', lw=0.5)
axs_1d[0].set_xlabel('Position (x)')
axs_1d[0].set_ylabel('Velocity (v_x)')
axs_1d[0].set_title('Phase Space Trajectory of 1D Motion')
axs_1d[0].grid(True, linestyle='--', alpha=0.7)

# 2. Normalized Energy Error Over Time (Log Scale)
axs_1d[1].plot(times, normalized_energy_error_1d_safe, color='orange', lw=1)
axs_1d[1].set_xlabel('Time')
axs_1d[1].set_ylabel('Energy Error (%)')
axs_1d[1].set_title('Normalized Energy Error Over Time (1D Motion)')
axs_1d[1].set_yscale('log')  # Set y-axis to logarithmic scale
axs_1d[1].grid(True, linestyle='--', alpha=0.7, which='both')

# 3. Absolute Percentage Error of Extrema Positions (Log Scale)
axs_1d[2].plot(orbit_numbers_max, absolute_percentage_diff_max_safe, 'b.-', markersize=3, label='Maxima % Error')
axs_1d[2].plot(orbit_numbers_min, absolute_percentage_diff_min_safe, 'r.-', markersize=3, label='Minima % Error')
axs_1d[2].set_xlabel('Orbit Number')
axs_1d[2].set_ylabel('Absolute Percentage Error (%)')
axs_1d[2].set_title('Absolute Percentage Error of Extrema Positions')
axs_1d[2].set_yscale('log')  # Set y-axis to logarithmic scale
axs_1d[2].legend()
axs_1d[2].grid(True, linestyle='--', alpha=0.7, which='both')

fig_1d.tight_layout()
plt.show()
