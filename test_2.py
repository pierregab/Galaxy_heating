import numpy as np
import matplotlib.pyplot as plt
import logging
import timeit
import os  # Import the os module for directory operations

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# Unit Definitions and Scaling Factors
# ============================================================

class System:
    """
    Base class for physical systems.

    Attributes:
        length_scale (float): Length scaling factor.
        mass_scale (float): Mass scaling factor.
        time_scale (float): Time scaling factor.
        velocity_scale (float): Velocity scaling factor.
        velocity_scale_kms (float): Velocity scaling factor in km/s.
        G (float): Gravitational constant in simulation units (normalized).
    """

    def __init__(self):
        # Logging simulation properties
        logging.info("Initializing system with the following properties:")

        # ------------------------------------------------------------
        # Physical Units (Astrophysical Units)
        # ------------------------------------------------------------
        # - Mass Unit (M0): 1 x 10^11 solar masses (Msun)
        # - Length Unit (R0): 1 kiloparsecs (kpc)
        # - Time Unit (T0): Derived from R0 and M0 using G
        # - Velocity Unit (V0): Derived from R0 and T0

        # Gravitational constant in physical units (kpc^3 Msun^-1 Myr^-2)
        self.G_physical = 4.498e-12  # G = 4.498 x 10^-12 kpc^3 Msun^-1 Myr^-2

        # Physical constants
        self.M0 = 1e11           # Mass unit in solar masses (Msun)
        self.R0 = 1              # Length unit in kiloparsecs (kpc)

        # Calculate the time unit (T0) in Myr
        self.T0 = np.sqrt(self.R0**3 / (self.G_physical * self.M0))  # Time unit in Myr

        # Calculate the velocity unit (V0) in kpc/Myr
        self.V0 = self.R0 / self.T0  # Velocity unit in kpc/Myr

        # Convert velocity unit to km/s (1 kpc/Myr = 977.8 km/s)
        self.V0_kms = self.V0 * 977.8  # Velocity unit in km/s

        # Log the scaling factors
        logging.info(f"Physical units:")
        logging.info(f"  Gravitational constant (G_physical): {self.G_physical} kpc^3 Msun^-1 Myr^-2")
        logging.info(f"  Mass unit (M0): {self.M0} Msun")
        logging.info(f"  Length unit (R0): {self.R0} kpc")
        logging.info(f"  Time unit (T0): {self.T0:.3f} Myr")
        logging.info(f"  Velocity unit (V0): {self.V0_kms:.3f} km/s")

        # ------------------------------------------------------------
        # Simulation Units (Dimensionless Units)
        # ------------------------------------------------------------
        # - Gravitational Constant (G): Set to 1 for normalization
        # - Mass (M): Set to 1 to normalize mass
        # - Radial Scale Length (a): Set to 2.0 (dimensionless)
        # - Vertical Scale Length (b): Set to 0.1 (dimensionless)

        # Normalize constants for simulation
        self.G = 1.0        # Gravitational constant (normalized)
        self.M = 1.0        # Mass (normalized)
        self.a = 2.0        # Radial scale length (normalized to R0)
        self.b = 0.1        # Vertical scale length (normalized)

        # Scaling Factors for conversion between simulation units and physical units
        self.length_scale = self.R0       # 1 simulation length unit = R0 kpc
        self.mass_scale = self.M0         # 1 simulation mass unit = M0 Msun
        self.time_scale = self.T0         # 1 simulation time unit = T0 Myr
        self.velocity_scale = self.V0     # 1 simulation velocity unit = V0 kpc/Myr
        self.velocity_scale_kms = self.V0_kms  # Velocity unit in km/s

        # Log simulation scaling factors
        logging.info(f"Simulation units (dimensionless):")
        logging.info(f"  Gravitational constant (G): {self.G} (normalized)")
        logging.info(f"  Mass (M): {self.M} (normalized)")
        logging.info(f"  Radial scale length (a): {self.a} (dimensionless)")
        logging.info(f"  Vertical scale length (b): {self.b} (dimensionless)")


class Galaxy(System):
    """
    Galaxy class representing the galactic potential.

    Methods:
        set_mass(mass): Set the mass of the galaxy.
        set_a(a): Set the radial scale length.
        set_b(b): Set the vertical scale length.
        potential(R, z): Compute the gravitational potential at (R, z).
        acceleration(pos): Compute the acceleration at position pos.
        initialize_stars(N, Rmax): Initialize N stars with positions drawn from uniform distributions.
    """

    def __init__(self, mass=1.0, a=2.0, b=None):
        super().__init__()
        self.M = mass        # Mass (normalized)
        self.a = a           # Radial scale length (normalized to R0)
        if b is None:
            self.b = 1 / 20 * self.a  # Vertical scale length (normalized)
        else:
            self.b = b

        # Log galaxy parameters
        logging.info(f"Galaxy parameters:")
        logging.info(f"  Mass (M): {self.M} (normalized)")
        logging.info(f"  Radial scale length (a): {self.a} (dimensionless)")
        logging.info(f"  Vertical scale length (b): {self.b} (dimensionless)")

        # Initialize list to hold Particle instances
        self.particles = []

    def set_mass(self, mass):
        """
        Set the mass of the galaxy.

        Parameters:
            mass (float): Mass of the galaxy (normalized).
        """
        self.M = mass
        logging.info(f"Galaxy mass set to {self.M} (normalized)")

    def set_a(self, a):
        """
        Set the radial scale length.

        Parameters:
            a (float): Radial scale length (normalized to R0).
        """
        self.a = a
        logging.info(f"Radial scale length (a) set to {self.a} (dimensionless)")

    def set_b(self, b):
        """
        Set the vertical scale length.

        Parameters:
            b (float): Vertical scale length (normalized).
        """
        self.b = b
        logging.info(f"Vertical scale length (b) set to {self.b} (dimensionless)")

    def potential(self, R, z):
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
        denom = np.sqrt(R**2 + (self.a + np.sqrt(z**2 + self.b**2))**2)
        return -self.G * self.M / denom

    def acceleration(self, pos):
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
        z_term = np.sqrt(z**2 + self.b**2)
        denom = (R**2 + (self.a + z_term)**2)**1.5
        ax = -self.G * self.M * x / denom
        ay = -self.G * self.M * y / denom
        az = -self.G * self.M * (self.a + z_term) * z / (z_term * denom)
        return np.array([ax, ay, az])

    def initialize_stars(self, N, Rmax):
        """
        Initialize N stars with positions drawn from uniform distributions.

        Parameters:
            N (int): Number of stars to initialize.
            Rmax (float): Maximum radial distance (dimensionless).
        """
        logging.info(f"Initializing {N} stars with uniform R in [0, {Rmax}] and phi in [0, 2π].")
        R = np.random.uniform(0, Rmax, N)  # Radial distances
        phi = np.random.uniform(0, 2 * np.pi, N)  # Angular positions

        # Convert polar coordinates to Cartesian coordinates (x, y)
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        z = np.zeros(N)  # All stars lie in the galactic plane

        # Calculate the circular velocity needed for a stable orbit for each star
        # v_circular = sqrt(G * M * R^2 / (R^2 + (a + b)^2)**1.5)
        v_circular = np.sqrt(self.G * self.M * R**2 / (R**2 + (self.a + self.b)**2)**1.5)  # [N]

        # Initialize velocities perpendicular to the radius vector for circular orbits
        # v_x = -v_circular * sin(phi), v_y = v_circular * cos(phi)
        v_x = -v_circular * np.sin(phi)
        v_y = v_circular * np.cos(phi)
        v_z = np.zeros(N)  # No vertical velocity

        # Create Particle instances
        for i in range(N):
            position = np.array([x[i], y[i], z[i]])  # [x, y, z]
            velocity = np.array([v_x[i], v_y[i], v_z[i]])  # [vx, vy, vz]
            particle = Particle(position, velocity)
            self.particles.append(particle)

        logging.info(f"Initialized {N} stars successfully.")


class Particle:
    """
    Particle class representing a particle in the simulation.

    Attributes:
        position (np.ndarray): Position vector [x, y, z].
        velocity (np.ndarray): Velocity vector [vx, vy, vz].
        energy (float): Energy of the particle.
        angular_momentum (float): Angular momentum of the particle.
    """

    def __init__(self, position, velocity):
        self.initial_position = np.copy(position)
        self.initial_velocity = np.copy(velocity)
        self.position = np.copy(position)
        self.velocity = np.copy(velocity)
        self.energy = None
        self.angular_momentum = None
        logging.info(f"Particle initialized with position {self.position} and velocity {self.velocity}.")

    def reset(self):
        """
        Reset the particle to its initial conditions.
        """
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        self.energy = None
        self.angular_momentum = None
        logging.info("Particle reset to initial conditions.")


class Integrator:
    """
    Integrator class containing integration methods.

    Methods:
        leapfrog(particles, galaxy, dt, steps): Perform leapfrog integration.
        rk4(particles, galaxy, dt, steps): Perform RK4 integration.
    """

    def leapfrog(self, particles, galaxy, dt, steps):
        """
        Leapfrog integrator for orbit simulation.

        Parameters:
            particles (list of Particle): List of Particle instances with initial positions and velocities.
            galaxy (Galaxy): Galaxy instance providing potential and acceleration.
            dt (float): Time step (dimensionless).
            steps (int): Number of integration steps.

        Returns:
            tuple:
                positions (np.ndarray): Positions over time [steps, N, 3].
                velocities (np.ndarray): Velocities over time [steps, N, 3].
                energies (np.ndarray): Total energy over time [steps, N].
                angular_momenta (np.ndarray): Angular momentum (Lz) over time [steps, N].
        """
        logging.info("Starting Leapfrog integration.")
        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N))

        # Initialize positions and velocities
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]

        # Calculate initial half-step velocities
        acc = galaxy.acceleration(pos.T).T  # [N, 3]
        vel_half = vel + 0.5 * dt * acc  # [N, 3]

        for i in range(steps):
            # Update positions
            pos += dt * vel_half  # [N, 3]
            positions[i] = pos

            # Calculate acceleration at new positions
            acc = galaxy.acceleration(pos.T).T  # [N, 3]

            # Update half-step velocities
            vel_half += dt * acc  # [N, 3]

            # Store full-step velocities
            vel_full = vel_half - 0.5 * dt * acc  # [N, 3]
            velocities[i] = vel_full

            # Compute kinetic and potential energy
            v = np.linalg.norm(vel_full, axis=1)  # [N]
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z)  # [N]
            kinetic_energy = 0.5 * v**2  # [N]
            energies[i] = kinetic_energy + potential_energy  # [N]

            # Compute angular momentum (Lz component)
            Lz = pos[:, 0] * vel_full[:, 1] - pos[:, 1] * vel_full[:, 0]  # [N]
            angular_momenta[i] = Lz

            # Log progress every 10%
            if (i+1) % (steps // 10) == 0:
                logging.info(f"Leapfrog integration progress: {100 * (i+1) / steps:.1f}%")

        logging.info("Leapfrog integration completed.")
        return positions, velocities, energies, angular_momenta

    def rk4(self, particles, galaxy, dt, steps):
        """
        Runge-Kutta 4th order integrator for orbit simulation.

        Parameters:
            particles (list of Particle): List of Particle instances with initial positions and velocities.
            galaxy (Galaxy): Galaxy instance providing potential and acceleration.
            dt (float): Time step (dimensionless).
            steps (int): Number of integration steps.

        Returns:
            tuple:
                positions (np.ndarray): Positions over time [steps, N, 3].
                velocities (np.ndarray): Velocities over time [steps, N, 3].
                energies (np.ndarray): Total energy over time [steps, N].
                angular_momenta (np.ndarray): Angular momentum (Lz) over time [steps, N].
        """
        logging.info("Starting RK4 integration.")
        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N))

        # Initialize positions and velocities
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]

        for i in range(steps):
            # k1
            acc1 = galaxy.acceleration(pos.T).T  # [N, 3]
            k1_vel = dt * acc1  # [N, 3]
            k1_pos = dt * vel  # [N, 3]

            # k2
            acc2 = galaxy.acceleration((pos + 0.5 * k1_pos).T).T  # [N, 3]
            k2_vel = dt * acc2  # [N, 3]
            k2_pos = dt * (vel + 0.5 * k1_vel)  # [N, 3]

            # k3
            acc3 = galaxy.acceleration((pos + 0.5 * k2_pos).T).T  # [N, 3]
            k3_vel = dt * acc3  # [N, 3]
            k3_pos = dt * (vel + 0.5 * k2_vel)  # [N, 3]

            # k4
            acc4 = galaxy.acceleration((pos + k3_pos).T).T  # [N, 3]
            k4_vel = dt * acc4  # [N, 3]
            k4_pos = dt * (vel + k3_vel)  # [N, 3]

            # Update positions and velocities
            pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6  # [N, 3]
            vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6  # [N, 3]

            positions[i] = pos
            velocities[i] = vel

            # Compute kinetic and potential energy
            v = np.linalg.norm(vel, axis=1)  # [N]
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z)  # [N]
            kinetic_energy = 0.5 * v**2  # [N]
            energies[i] = kinetic_energy + potential_energy  # [N]

            # Compute angular momentum (Lz component)
            Lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]  # [N]
            angular_momenta[i] = Lz

            # Log progress every 10%
            if (i+1) % (steps // 10) == 0:
                logging.info(f"RK4 integration progress: {100 * (i+1) / steps:.1f}%")

        logging.info("RK4 integration completed.")
        return positions, velocities, energies, angular_momenta


class Simulation(System):
    """
    Simulation class to set up and run the simulation.

    Methods:
        run(): Run the simulation.
        plot_energy_error(): Plot the energy error over time.
        plot_angular_momentum_error(): Plot the angular momentum error over time.
        plot_trajectories(): Plot the orbit trajectories.
        plot_execution_time(): Plot execution times per step.
    """

    def __init__(self, galaxy, dt=0.05, t_max=250.0):
        super().__init__()
        self.galaxy = galaxy
        self.dt = dt
        self.t_max = t_max
        self.steps = int(self.t_max / self.dt)
        self.times = np.linspace(0, self.t_max, self.steps)
        self.integrator = Integrator()
        self.positions_lf = None
        self.velocities_lf = None
        self.energies_lf = None
        self.angular_momenta_lf = None
        self.positions_rk4 = None
        self.velocities_rk4 = None
        self.energies_rk4 = None
        self.angular_momenta_rk4 = None
        self.execution_times = {}

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

    def run(self):
        """
        Run the simulation using both Leapfrog and RK4 integrators.
        """
        logging.info("Starting the simulation.")

        # Leapfrog Integration Timing
        def run_leapfrog():
            return self.integrator.leapfrog(self.galaxy.particles, self.galaxy, self.dt, self.steps)

        start_time = timeit.default_timer()
        self.positions_lf, self.velocities_lf, self.energies_lf, self.angular_momenta_lf = run_leapfrog()
        total_time_lf = timeit.default_timer() - start_time
        average_time_lf = total_time_lf / self.steps
        logging.info(f"Leapfrog integration took {total_time_lf:.3f} seconds in total.")
        logging.info(f"Average time per step (Leapfrog): {average_time_lf*1e3:.6f} ms.")
        self.execution_times['Leapfrog'] = average_time_lf * 1e3  # in ms

        # RK4 Integration Timing
        def run_rk4():
            return self.integrator.rk4(self.galaxy.particles, self.galaxy, self.dt, self.steps)

        start_time = timeit.default_timer()
        self.positions_rk4, self.velocities_rk4, self.energies_rk4, self.angular_momenta_rk4 = run_rk4()
        total_time_rk4 = timeit.default_timer() - start_time
        average_time_rk4 = total_time_rk4 / self.steps
        logging.info(f"RK4 integration took {total_time_rk4:.3f} seconds in total.")
        logging.info(f"Average time per step (RK4): {average_time_rk4*1e3:.6f} ms.")
        self.execution_times['RK4'] = average_time_rk4 * 1e3  # in ms

        logging.info("Simulation completed.")

    def plot_trajectories(self):
        """
        Plot the orbit trajectories in the xy-plane.
        """
        logging.info("Generating orbit trajectory plot.")

        N = len(self.galaxy.particles)

        # Leapfrog: Convert positions to physical units
        positions_lf_physical = self.positions_lf * self.length_scale  # [steps, N, 3]

        # RK4: Convert positions to physical units
        positions_rk4_physical = self.positions_rk4 * self.length_scale  # [steps, N, 3]

        plt.figure(figsize=(12, 10))
        for i in range(N):
            plt.plot(positions_lf_physical[:, i, 0], positions_lf_physical[:, i, 1],
                     label='Leapfrog' if i == 0 else "", linewidth=0.5, alpha=0.7)
            plt.plot(positions_rk4_physical[:, i, 0], positions_rk4_physical[:, i, 1],
                     label='RK4' if i == 0 else "", linestyle='--', linewidth=0.5, alpha=0.7)

        plt.xlabel('x (kpc)', fontsize=14)
        plt.ylabel('y (kpc)', fontsize=14)
        plt.title('Orbit Trajectories in Galactic Potential', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'orbit_xy_plane.png'))
        logging.info(f"Orbit trajectory plot saved to '{self.results_dir}/orbit_xy_plane.png'.")

    def plot_energy_error(self):
        """
        Plot the energy conservation error over time.
        """
        logging.info("Generating energy conservation error plot.")

        # Time array in physical units
        times_physical = self.times * self.time_scale  # Time in Myr

        # Convert energies to physical units
        energies_lf_physical = self.energies_lf * self.velocity_scale_kms**2  # [steps, N]
        energies_rk4_physical = self.energies_rk4 * self.velocity_scale_kms**2  # [steps, N]

        # Compute energy errors
        E0_lf = energies_lf_physical[0]  # [N]
        E_error_lf = (energies_lf_physical - E0_lf) / np.abs(E0_lf)  # [steps, N]

        E0_rk4 = energies_rk4_physical[0]  # [N]
        E_error_rk4 = (energies_rk4_physical - E0_rk4) / np.abs(E0_rk4)  # [steps, N]

        # Compute average energy error across all stars
        avg_E_error_lf = np.mean(np.abs(E_error_lf), axis=1)  # [steps]
        avg_E_error_rk4 = np.mean(np.abs(E_error_rk4), axis=1)  # [steps]

        plt.figure(figsize=(12, 10))
        plt.plot(times_physical, avg_E_error_lf, label='Leapfrog', linewidth=1)
        plt.plot(times_physical, avg_E_error_rk4, label='RK4', linestyle='--', linewidth=1)
        plt.xlabel('Time (Myr)', fontsize=14)
        plt.ylabel('Relative Energy Error', fontsize=14)
        plt.title('Energy Conservation Error Over Time', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'energy_error.png'))
        logging.info(f"Energy conservation error plot saved to '{self.results_dir}/energy_error.png'.")

    def plot_angular_momentum_error(self):
        """
        Plot the angular momentum conservation error over time.
        """
        logging.info("Generating angular momentum conservation error plot.")

        # Time array in physical units
        times_physical = self.times * self.time_scale  # Time in Myr

        # Convert angular momenta to physical units
        angular_momenta_lf_physical = self.angular_momenta_lf * self.length_scale * self.velocity_scale_kms  # [steps, N]
        angular_momenta_rk4_physical = self.angular_momenta_rk4 * self.length_scale * self.velocity_scale_kms  # [steps, N]

        # Compute angular momentum errors
        L0_lf = angular_momenta_lf_physical[0]  # [N]
        L_error_lf = (angular_momenta_lf_physical - L0_lf) / np.abs(L0_lf)  # [steps, N]

        L0_rk4 = angular_momenta_rk4_physical[0]  # [N]
        L_error_rk4 = (angular_momenta_rk4_physical - L0_rk4) / np.abs(L0_rk4)  # [steps, N]

        # Compute average angular momentum error across all stars
        avg_L_error_lf = np.mean(np.abs(L_error_lf), axis=1)  # [steps]
        avg_L_error_rk4 = np.mean(np.abs(L_error_rk4), axis=1)  # [steps]

        plt.figure(figsize=(12, 10))
        plt.plot(times_physical, avg_L_error_lf, label='Leapfrog', linewidth=1)
        plt.plot(times_physical, avg_L_error_rk4, label='RK4', linestyle='--', linewidth=1)
        plt.xlabel('Time (Myr)', fontsize=14)
        plt.ylabel('Relative Angular Momentum Error', fontsize=14)
        plt.title('Angular Momentum Conservation Error Over Time', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'angular_momentum_error.png'))
        logging.info(f"Angular momentum conservation error plot saved to '{self.results_dir}/angular_momentum_error.png'.")

    def plot_execution_time(self):
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
        logging.info(f"Execution time comparison plot saved to '{self.results_dir}/execution_times.png'.")


def main():
    # ============================================================
    # Initial Conditions (Dimensionless Units)
    # ============================================================

    # Number of stars
    N_stars = 10000  # You can adjust this number as needed

    # Maximum radial distance (Rmax) in dimensionless units
    Rmax = 10.0  # Adjust based on the simulation needs

    # Logging simulation properties
    logging.info("Starting the simulation with the following properties:")

    # Create Galaxy instance
    galaxy = Galaxy(mass=1.0, a=2.0, b=0.1)

    # Initialize stars with uniform R and phi distributions
    galaxy.initialize_stars(N=N_stars, Rmax=Rmax)

    # Create Simulation instance
    simulation = Simulation(galaxy=galaxy, dt=0.05, t_max=250.0)

    # Run the simulation
    simulation.run()

    # ============================================================
    # Generate Plots
    # ============================================================

    # Generate plots using the Simulation class methods
    simulation.plot_trajectories()
    simulation.plot_energy_error()
    simulation.plot_angular_momentum_error()
    simulation.plot_execution_time()


if __name__ == '__main__':
    main()

