import numpy as np
import matplotlib.pyplot as plt
import logging
import timeit

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# Unit Definitions and Scaling Factors
# ============================================================

# Logging simulation properties
logging.info("Starting the simulation with the following properties:")

# ------------------------------------------------------------
# Physical Units (Astrophysical Units)
# ------------------------------------------------------------
# - Mass Unit (M0): 1 x 10^11 solar masses (Msun)
# - Length Unit (R0): 2.5 kiloparsecs (kpc)
# - Time Unit (T0): Derived from R0 and M0 using G
# - Velocity Unit (V0): Derived from R0 and T0

# Gravitational constant in physical units (kpc^3 Msun^-1 Myr^-2)
G_physical = 4.498e-12  # G = 4.498 x 10^-12 kpc^3 Msun^-1 Myr^-2

# Physical constants
M0 = 1e11           # Mass unit in solar masses (Msun)
R0 = 1              # Length unit in kiloparsecs (kpc)

# Calculate the time unit (T0) in Myr
T0 = np.sqrt(R0**3 / (G_physical * M0))  # Time unit in Myr

# Calculate the velocity unit (V0) in kpc/Myr
V0 = R0 / T0  # Velocity unit in kpc/Myr

# Convert velocity unit to km/s (1 kpc/Myr = 977.8 km/s)
V0_kms = V0 * 977.8  # Velocity unit in km/s

# Log the scaling factors
logging.info(f"Physical units:")
logging.info(f"  Gravitational constant (G_physical): {G_physical} kpc^3 Msun^-1 Myr^-2")
logging.info(f"  Mass unit (M0): {M0} Msun")
logging.info(f"  Length unit (R0): {R0} kpc")
logging.info(f"  Time unit (T0): {T0:.3f} Myr")
logging.info(f"  Velocity unit (V0): {V0_kms:.3f} km/s")

# ------------------------------------------------------------
# Simulation Units (Dimensionless Units)
# ------------------------------------------------------------
# - Gravitational Constant (G): Set to 1 for normalization
# - Mass (M): Set to 1 to normalize mass
# - Radial Scale Length (a): Set to 1 (dimensionless)
# - Vertical Scale Length (b): Set to 0.05 (dimensionless)

# Normalize constants for simulation
G = 1.0        # Gravitational constant (normalized)
M = 1.0        # Mass (normalized)
a = 2.0        # Radial scale length (normalized to R0)
b = 1/20 * a       # Vertical scale length (normalized)

# Scaling Factors for conversion between simulation units and physical units
length_scale = R0       # 1 simulation length unit = R0 kpc
mass_scale = M0         # 1 simulation mass unit = M0 Msun
time_scale = T0         # 1 simulation time unit = T0 Myr
velocity_scale = V0     # 1 simulation velocity unit = V0 kpc/Myr
velocity_scale_kms = V0_kms  # Velocity unit in km/s

# ============================================================
# Time Parameters (Dimensionless Units)
# ============================================================

dt = 0.001     # Time step (dimensionless)
t_max = 250.0   # Total simulation time (dimensionless)
steps = int(t_max / dt)  # Number of integration steps

logging.info(f"Simulation time parameters:")
logging.info(f"  Time step (dt): {dt} (dimensionless)")
logging.info(f"  Total simulation time (t_max): {t_max} (dimensionless)")
logging.info(f"  Number of steps: {steps}")

# ============================================================
# Potential and Acceleration Functions
# ============================================================

def potential(R, z):
    """
    Compute the gravitational potential at a given (R, z).

    Parameters:
    R : float or np.ndarray
        Radial distance from the galactic center (dimensionless).
    z : float or np.ndarray
        Vertical distance from the galactic plane (dimensionless).

    Returns:
    float or np.ndarray
        Gravitational potential at the specified location (dimensionless).
    """
    denom = np.sqrt(R**2 + (a + np.sqrt(z**2 + b**2))**2)
    return -G * M / denom

def acceleration(pos):
    """
    Compute the acceleration vector at a given position in the galaxy.

    Parameters:
    pos : np.ndarray
        Position vector [x, y, z] in dimensionless units.

    Returns:
    np.ndarray
        Acceleration vector [ax, ay, az] in dimensionless units.
    """
    x, y, z = pos
    R = np.sqrt(x**2 + y**2)
    z_term = np.sqrt(z**2 + b**2)
    denom = (R**2 + (a + z_term)**2)**1.5
    ax = -G * M * x / denom
    ay = -G * M * y / denom
    az = -G * M * (a + z_term) * z / (z_term * denom)
    return np.array([ax, ay, az])

# ============================================================
# Integrator Functions
# ============================================================

def leapfrog_integrator(pos0, vel0, dt, steps):
    """
    Leapfrog integrator for orbit simulation.

    Parameters:
    pos0 : np.ndarray
        Initial position vector [x, y, z] (dimensionless).
    vel0 : np.ndarray
        Initial velocity vector [vx, vy, vz] (dimensionless).
    dt : float
        Time step (dimensionless).
    steps : int
        Number of integration steps.

    Returns:
    tuple:
        positions (np.ndarray): Positions over time.
        velocities (np.ndarray): Velocities over time.
        energies (np.ndarray): Total energy over time.
        angular_momenta (np.ndarray): Angular momentum (Lz) over time.
    """
    logging.info("Starting Leapfrog integration.")
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))
    energies = np.zeros(steps)
    angular_momenta = np.zeros(steps)

    pos = np.copy(pos0)
    vel = np.copy(vel0)
    vel_half = vel + 0.5 * dt * acceleration(pos)

    for i in range(steps):
        # Update position
        pos += dt * vel_half
        positions[i] = pos

        # Calculate acceleration at the new position
        acc = acceleration(pos)

        # Update velocity at the next half-step
        vel_half += dt * acc

        # Store the full-step velocity (average between half-steps)
        velocities[i] = vel_half - 0.5 * dt * acc

        # Compute kinetic and potential energy
        v = np.linalg.norm(velocities[i])
        R = np.sqrt(pos[0]**2 + pos[1]**2)
        z = pos[2]
        energies[i] = 0.5 * v**2 + potential(R, z)

        # Compute angular momentum (Lz component)
        angular_momenta[i] = pos[0] * velocities[i][1] - pos[1] * velocities[i][0]

    logging.info("Leapfrog integration completed.")
    return positions, velocities, energies, angular_momenta

def rk4_integrator(pos0, vel0, dt, steps):
    """
    Runge-Kutta 4th order integrator for orbit simulation.

    Parameters:
    pos0 : np.ndarray
        Initial position vector [x, y, z] (dimensionless).
    vel0 : np.ndarray
        Initial velocity vector [vx, vy, vz] (dimensionless).
    dt : float
        Time step (dimensionless).
    steps : int
        Number of integration steps.

    Returns:
    tuple:
        positions (np.ndarray): Positions over time.
        velocities (np.ndarray): Velocities over time.
        energies (np.ndarray): Total energy over time.
        angular_momenta (np.ndarray): Angular momentum (Lz) over time.
    """
    logging.info("Starting RK4 integration.")
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))
    energies = np.zeros(steps)
    angular_momenta = np.zeros(steps)

    pos = np.copy(pos0)
    vel = np.copy(vel0)

    for i in range(steps):
        # k1
        acc1 = acceleration(pos)
        k1_vel = dt * acc1
        k1_pos = dt * vel

        # k2
        acc2 = acceleration(pos + 0.5 * k1_pos)
        k2_vel = dt * acc2
        k2_pos = dt * (vel + 0.5 * k1_vel)

        # k3
        acc3 = acceleration(pos + 0.5 * k2_pos)
        k3_vel = dt * acc3
        k3_pos = dt * (vel + 0.5 * k2_vel)

        # k4
        acc4 = acceleration(pos + k3_pos)
        k4_vel = dt * acc4
        k4_pos = dt * (vel + k3_vel)

        # Update position and velocity
        pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
        vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6

        positions[i] = pos
        velocities[i] = vel

        # Compute kinetic and potential energy
        v = np.linalg.norm(vel)
        R = np.sqrt(pos[0]**2 + pos[1]**2)
        z = pos[2]
        energies[i] = 0.5 * v**2 + potential(R, z)

        # Compute angular momentum (Lz component)
        angular_momenta[i] = pos[0] * vel[1] - pos[1] * vel[0]

    logging.info("RK4 integration completed.")
    return positions, velocities, energies, angular_momenta

# ============================================================
# Initial Conditions (Dimensionless Units)
# ============================================================

# Initial radial position (dimensionless)
r0 = 8.0  # Set to 2 x scale length 'a' for this example

# Calculate the circular velocity needed for a stable orbit
R = r0
z = 0.0
v_circular = np.sqrt(G * M * R**2 / (R**2 + (a + np.sqrt(z**2 + b**2))**2)**1.5)

# Initial position and velocity vectors (dimensionless)
pos_initial = np.array([r0, 0.0, 0.0])       # Starting at x = r0, y = 0, z = 0
vel_initial = np.array([0.0, v_circular, 0.0])  # Velocity in the y-direction for circular motion

logging.info(f"Initial conditions:")
logging.info(f"  Initial position: {pos_initial} (dimensionless)")
logging.info(f"  Initial velocity: {vel_initial} (dimensionless)")

# ============================================================
# Perform Simulations and Measure Execution Time
# ============================================================

# Leapfrog Integration Timing
def run_leapfrog():
    positions_lf, velocities_lf, energies_lf, angular_momenta_lf = leapfrog_integrator(
        pos_initial, vel_initial, dt, steps)
    return positions_lf, velocities_lf, energies_lf, angular_momenta_lf

# Measure total time for Leapfrog integration
start_time = timeit.default_timer()
positions_lf, velocities_lf, energies_lf, angular_momenta_lf = run_leapfrog()
total_time_lf = timeit.default_timer() - start_time
average_time_lf = total_time_lf / steps
logging.info(f"Leapfrog integration took {total_time_lf:.3f} seconds in total.")
logging.info(f"Average time per step (Leapfrog): {average_time_lf*1e3:.6f} ms.")

# RK4 Integration Timing
def run_rk4():
    positions_rk4, velocities_rk4, energies_rk4, angular_momenta_rk4 = rk4_integrator(
        pos_initial, vel_initial, dt, steps)
    return positions_rk4, velocities_rk4, energies_rk4, angular_momenta_rk4

# Measure total time for RK4 integration
start_time = timeit.default_timer()
positions_rk4, velocities_rk4, energies_rk4, angular_momenta_rk4 = run_rk4()
total_time_rk4 = timeit.default_timer() - start_time
average_time_rk4 = total_time_rk4 / steps
logging.info(f"RK4 integration took {total_time_rk4:.3f} seconds in total.")
logging.info(f"Average time per step (RK4): {average_time_rk4*1e3:.6f} ms.")

# ============================================================
# Conversion to Physical Units
# ============================================================

# Time array (dimensionless and physical)
times = np.linspace(0, t_max, steps)
times_physical = times * time_scale  # Time in Myr

# Leapfrog: Convert positions and velocities to physical units
positions_lf_physical = positions_lf * length_scale  # Positions in kpc
velocities_lf_physical = velocities_lf * velocity_scale_kms  # Velocities in km/s
energies_lf_physical = energies_lf * velocity_scale_kms**2  # Energy per unit mass in (km/s)^2
angular_momenta_lf_physical = angular_momenta_lf * length_scale * velocity_scale_kms

# RK4: Convert positions and velocities to physical units
positions_rk4_physical = positions_rk4 * length_scale  # Positions in kpc
velocities_rk4_physical = velocities_rk4 * velocity_scale_kms  # Velocities in km/s
energies_rk4_physical = energies_rk4 * velocity_scale_kms**2  # Energy per unit mass in (km/s)^2
angular_momenta_rk4_physical = angular_momenta_rk4 * length_scale * velocity_scale_kms

# ============================================================
# Compute Energy and Angular Momentum Errors
# ============================================================

# For Leapfrog Integrator
E0_lf = energies_lf_physical[0]
L0_lf = angular_momenta_lf_physical[0]

E_error_lf = (energies_lf_physical - E0_lf) / np.abs(E0_lf)
L_error_lf = (angular_momenta_lf_physical - L0_lf) / np.abs(L0_lf)

# For RK4 Integrator
E0_rk4 = energies_rk4_physical[0]
L0_rk4 = angular_momenta_rk4_physical[0]

E_error_rk4 = (energies_rk4_physical - E0_rk4) / np.abs(E0_rk4)
L_error_rk4 = (angular_momenta_rk4_physical - L0_rk4) / np.abs(L0_rk4)

# ============================================================
# Plotting Results
# ============================================================

logging.info("Generating plots.")

# Plot the orbit in the xy-plane (physical units)
plt.figure(figsize=(12, 10))
plt.plot(positions_lf_physical[:, 0], positions_lf_physical[:, 1], label='Leapfrog', linewidth=1)
plt.plot(positions_rk4_physical[:, 0], positions_rk4_physical[:, 1], label='RK4', linestyle='--', linewidth=1)
plt.xlabel('x (kpc)', fontsize=14)
plt.ylabel('y (kpc)', fontsize=14)
plt.title('Orbit Trajectory in Galactic Potential', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Plot the energy errors over time to check energy conservation
plt.figure(figsize=(12, 10))
plt.plot(times_physical, E_error_lf, label='Leapfrog', linewidth=1)
plt.plot(times_physical, E_error_rk4, label='RK4', linestyle='--', linewidth=1)
plt.xlabel('Time (Myr)', fontsize=14)
plt.ylabel('Relative Energy Error', fontsize=14)
plt.title('Energy Conservation Error Over Time', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot angular momentum errors over time to check conservation
plt.figure(figsize=(12, 10))
plt.plot(times_physical, L_error_lf, label='Leapfrog', linewidth=1)
plt.plot(times_physical, L_error_rk4, label='RK4', linestyle='--', linewidth=1)
plt.xlabel('Time (Myr)', fontsize=14)
plt.ylabel('Relative Angular Momentum Error', fontsize=14)
plt.title('Angular Momentum Conservation Error Over Time', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot execution times per step
plt.figure(figsize=(10, 8))
methods = ['Leapfrog', 'RK4']
times_exec = [average_time_lf * 1e3, average_time_rk4 * 1e3]  # Convert to milliseconds
plt.bar(methods, times_exec, color=['blue', 'orange'])
plt.ylabel('Average Time per Step (ms)', fontsize=14)
plt.title('Integrator Execution Time Comparison', fontsize=16)
for i, v in enumerate(times_exec):
    plt.text(i, v + 0.01, f"{v:.3f} ms", ha='center', fontsize=12)
plt.tight_layout()
plt.show()
