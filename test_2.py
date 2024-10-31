import numpy as np
import matplotlib.pyplot as plt
import logging
import timeit
import os  # Import the os module for directory operations

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Change dpi setting for higher resolution plots
plt.rcParams['figure.dpi'] = 150

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
        initialize_stars(N, Rmax): Initialize N stars with positions drawn from the Schwarzschild distribution.
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

    def dPhidr(self, R, z=0):
        """
        Compute the derivative of the potential with respect to R at given R and z.

        Parameters:
            R : float or np.ndarray
                Radial distance(s) (dimensionless).
            z : float or np.ndarray
                Vertical distance(s) (dimensionless), default is 0.

        Returns:
            float or np.ndarray
                dPhi/dR evaluated at each (R, z).
        """
        a = self.a
        b = self.b
        denom = np.sqrt(R**2 + (a + np.sqrt(z**2 + b**2))**2)
        dPhi_dR = -self.G * self.M * R / denom**3
        return dPhi_dR

    def omega(self, R):
        """
        Compute the angular frequency Omega at radius R.

        Parameters:
            R : float or np.ndarray
                Radial distance(s) (dimensionless).

        Returns:
            float or np.ndarray
                Angular frequency Omega at each R.
        """
        dPhi_dR = self.dPhidr(R)
        Omega = np.sqrt(R * (-dPhi_dR)) / R
        return Omega

    def kappa(self, R):
        """
        Compute the epicyclic frequency kappa at radius R.

        Parameters:
            R : float or np.ndarray
                Radial distance(s) (dimensionless).

        Returns:
            float or np.ndarray
                Epicyclic frequency kappa at each R.
        """
        a = self.a
        b = self.b
        denom = R**2 + (a + b)**2
        Omega = self.omega(R)
        # Correct formula for kappa^2 in terms of potential derivatives
        dPhidr = self.dPhidr(R)
        d2Phidr2 = ( -self.G * self.M / (denom)**3 ) * (1 - 3 * R**2 / denom)
        kappa_squared = 4 * Omega**2 + R * d2Phidr2
        kappa_squared = np.maximum(kappa_squared, 0)  # Avoid negative values due to numerical errors
        return np.sqrt(kappa_squared)

    def initialize_stars(self, N, Rmax, alpha=0.05, q=0.6, max_iterations=100):
        """
        Initialize N stars with positions and velocities drawn from the Schwarzschild velocity distribution function.

        Parameters:
            N (int): Number of stars to initialize.
            Rmax (float): Maximum radial distance (dimensionless).
            alpha (float): Small parameter for radial displacement (default: 0.05).
            q (float): Ratio of sigma_z to sigma_R (default: 0.6).
            max_iterations (int): Maximum number of velocity regeneration iterations (default: 100).
        """
        logging.info(f"Initializing {N} stars using the Schwarzschild velocity distribution function.")

        # Generate positions
        R_c = np.random.uniform(0.5 * Rmax, Rmax, N)  # Reference radii
        phi = np.random.uniform(0, 2 * np.pi, N)      # Angular positions

        # Small radial displacements x = R - R_c
        x = np.random.uniform(-alpha * R_c, alpha * R_c, N)  # Radial displacements
        R = R_c + x                                         # Actual radial positions

        # Ensure R is positive
        R = np.abs(R)

        # Positions in Cartesian coordinates
        x_pos = R * np.cos(phi)
        y_pos = R * np.sin(phi)
        z_pos = np.zeros(N)  # All stars lie in the galactic plane

        # Compute circular velocity at R
        v_c = np.sqrt(R * (-self.dPhidr(R)))  # Circular velocity at R

        # Angular frequency Omega
        Omega = self.omega(R)

        # Epicyclic frequency kappa
        kappa = self.kappa(R)
        kappa_squared = kappa**2

        # Gamma parameter
        gamma = 2 * Omega / kappa

        # Corrected Radial velocity dispersion sigma_R
        sigma_R_squared = (4/3) * kappa_squared * (alpha**2) * R_c**2
        sigma_R = np.sqrt(sigma_R_squared)

        # Vertical velocity dispersion sigma_z
        sigma_z = q * sigma_R  # Simple proportionality

        # Initialize arrays for velocities
        v_R = np.zeros(N)
        v_phi = np.zeros(N)
        v_z = np.zeros(N)

        iterations = 0
        unbound = np.ones(N, dtype=bool)  # Initially, all stars are unbound to enter the loop

        while np.any(unbound) and iterations < max_iterations:
            logging.info(f"Velocity generation iteration {iterations + 1}")

            # Generate velocities for unbound stars
            idx_unbound = np.where(unbound)[0]
            num_unbound = len(idx_unbound)

            if num_unbound == 0:
                break  # All stars are bound

            v_R_new = np.random.normal(0, sigma_R[idx_unbound])
            v_z_new = np.random.normal(0, sigma_z[idx_unbound])
            v_phi_new = v_c[idx_unbound] - gamma[idx_unbound] * v_R_new

            # Compute angular momentum L_z for these stars
            L_z_new = R[idx_unbound] * v_phi_new

            # Compute total mechanical energy per unit mass
            kinetic_energy_new = 0.5 * (v_R_new**2 + v_z_new**2)        # Radial and vertical kinetic energy
            rotational_energy_new = 0.5 * (L_z_new**2) / R[idx_unbound]**2      # Rotational kinetic energy
            potential_energy_new = self.potential(R[idx_unbound], z_pos[idx_unbound])     # Gravitational potential
            E_total_new = kinetic_energy_new + rotational_energy_new + potential_energy_new  # [N]

            # Compute escape speed squared
            escape_speed_squared = -2 * potential_energy_new

            # Total speed squared
            total_speed_squared = v_R_new**2 + v_phi_new**2 + v_z_new**2

            # Identify stars with total energy >= 0 or speed exceeding escape speed
            unbound_new = (E_total_new >= 0) | (total_speed_squared >= escape_speed_squared)
            unbound[idx_unbound] = unbound_new

            # Update velocities for bound stars
            bound_indices = idx_unbound[~unbound_new]
            v_R[bound_indices] = v_R_new[~unbound_new]
            v_z[bound_indices] = v_z_new[~unbound_new]
            v_phi[bound_indices] = v_phi_new[~unbound_new]

            logging.info(f"  {np.sum(~unbound_new)} stars bound in this iteration.")
            iterations += 1

        if iterations == max_iterations and np.any(unbound):
            num_remaining = np.sum(unbound)
            logging.warning(f"Maximum iterations reached. {num_remaining} stars remain unbound.")
            # Remove unbound stars
            idx_bound = np.where(~unbound)[0]
            R = R[idx_bound]
            phi = phi[idx_bound]
            x_pos = x_pos[idx_bound]
            y_pos = y_pos[idx_bound]
            z_pos = z_pos[idx_bound]
            v_R = v_R[idx_bound]
            v_phi = v_phi[idx_bound]
            v_z = v_z[idx_bound]
            N = len(idx_bound)
            logging.info(f"Proceeding with {N} bound stars.")
        else:
            logging.info(f"All {N} stars initialized as bound orbits.")

        # Convert velocities to Cartesian coordinates
        v_x = v_R * np.cos(phi) - v_phi * np.sin(phi)
        v_y = v_R * np.sin(phi) + v_phi * np.cos(phi)

        # Create Particle instances
        for i in range(N):
            position = np.array([x_pos[i], y_pos[i], z_pos[i]])  # [x, y, z]
            velocity = np.array([v_x[i], v_y[i], v_z[i]])        # [vx, vy, vz]
            particle = Particle(position, velocity)
            self.particles.append(particle)

        logging.info(f"Initialization complete with {N} particles.")

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
        logging.debug(f"Particle initialized with position {self.position} and velocity {self.velocity}.")

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
        acc = np.array([galaxy.acceleration(particle.position) for particle in particles])  # [N, 3]
        vel_half = vel + 0.5 * dt * acc  # [N, 3]

        for i in range(steps):
            # Update positions
            pos += dt * vel_half  # [N, 3]
            positions[i] = pos

            # Calculate acceleration at new positions
            acc = np.array([galaxy.acceleration(p) for p in pos])  # [N, 3]

            # Update half-step velocities
            vel_half += dt * acc  # [N, 3]

            # Store full-step velocities
            vel_full = vel_half - 0.5 * dt * acc  # [N, 3]
            velocities[i] = vel_full

            # Compute kinetic and potential energy
            v = np.linalg.norm(vel_full, axis=1)  # [N]
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z)     # [N]
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
            acc1 = np.array([galaxy.acceleration(p) for p in pos])  # [N, 3]
            k1_vel = dt * acc1  # [N, 3]
            k1_pos = dt * vel  # [N, 3]

            # k2
            acc2 = np.array([galaxy.acceleration(p) for p in pos + 0.5 * k1_pos])  # [N, 3]
            k2_vel = dt * acc2  # [N, 3]
            k2_pos = dt * (vel + 0.5 * k1_vel)  # [N, 3]

            # k3
            acc3 = np.array([galaxy.acceleration(p) for p in pos + 0.5 * k2_pos])  # [N, 3]
            k3_vel = dt * acc3  # [N, 3]
            k3_pos = dt * (vel + 0.5 * k2_vel)  # [N, 3]

            # k4
            acc4 = np.array([galaxy.acceleration(p) for p in pos + k3_pos])  # [N, 3]
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
            potential_energy = galaxy.potential(R, z)     # [N]
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

    def plot_trajectories(self, subset=100):
        """
        Plot the orbit trajectories in the xy-plane and xz-plane separately for each integrator.

        Parameters:
            subset (int): Number of stars to plot for clarity. Defaults to 100.
        """
        logging.info("Generating orbit trajectory plots.")

        N = len(self.galaxy.particles)
        subset = min(subset, N)  # Ensure subset does not exceed total number of stars
        indices = np.random.choice(N, subset, replace=False)  # Randomly select stars to plot

        # Leapfrog Orbits
        plt.figure(figsize=(12, 6))

        # x-y plot
        plt.subplot(1, 2, 1)
        for i in indices:
            plt.plot(self.positions_lf[:, i, 0] * self.length_scale,
                     self.positions_lf[:, i, 1] * self.length_scale,
                     linewidth=0.5, alpha=0.7)
        plt.xlabel('x (kpc)', fontsize=12)
        plt.ylabel('y (kpc)', fontsize=12)
        plt.title('Leapfrog: Orbit Trajectories in x-y Plane', fontsize=14)
        plt.grid(True)
        plt.axis('equal')

        # x-z plot
        plt.subplot(1, 2, 2)
        for i in indices:
            plt.plot(self.positions_lf[:, i, 0] * self.length_scale,
                     self.positions_lf[:, i, 2] * self.length_scale,
                     linewidth=0.5, alpha=0.7)
        plt.xlabel('x (kpc)', fontsize=12)
        plt.ylabel('z (kpc)', fontsize=12)
        plt.title('Leapfrog: Orbit Trajectories in x-z Plane', fontsize=14)
        plt.grid(True)
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'orbit_leapfrog.png'))
        plt.close()
        logging.info(f"Leapfrog orbit trajectories plots saved to '{self.results_dir}/orbit_leapfrog.png'.")

        # RK4 Orbits
        plt.figure(figsize=(12, 6))

        # x-y plot
        plt.subplot(1, 2, 1)
        for i in indices:
            plt.plot(self.positions_rk4[:, i, 0] * self.length_scale,
                     self.positions_rk4[:, i, 1] * self.length_scale,
                     linewidth=0.5, alpha=0.7)
        plt.xlabel('x (kpc)', fontsize=12)
        plt.ylabel('y (kpc)', fontsize=12)
        plt.title('RK4: Orbit Trajectories in x-y Plane', fontsize=14)
        plt.grid(True)
        plt.axis('equal')

        # x-z plot
        plt.subplot(1, 2, 2)
        for i in indices:
            plt.plot(self.positions_rk4[:, i, 0] * self.length_scale,
                     self.positions_rk4[:, i, 2] * self.length_scale,
                     linewidth=0.5, alpha=0.7)
        plt.xlabel('x (kpc)', fontsize=12)
        plt.ylabel('z (kpc)', fontsize=12)
        plt.title('RK4: Orbit Trajectories in x-z Plane', fontsize=14)
        plt.grid(True)
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'orbit_rk4.png'))
        plt.close()
        logging.info(f"RK4 orbit trajectories plots saved to '{self.results_dir}/orbit_rk4.png'.")

    def plot_energy_error(self):
        """
        Plot the energy conservation error over time.
        """
        logging.info("Generating energy conservation error plot.")

        # Time array in physical units
        times_physical = self.times * self.time_scale  # Time in Myr

        # Compute total energy for Leapfrog
        E0_lf = self.energies_lf[0]  # [N]
        E_error_lf = (self.energies_lf - E0_lf) / np.abs(E0_lf)  # [steps, N]

        # Compute total energy for RK4
        E0_rk4 = self.energies_rk4[0]  # [N]
        E_error_rk4 = (self.energies_rk4 - E0_rk4) / np.abs(E0_rk4)  # [steps, N]

        # Compute average energy error across all stars
        avg_E_error_lf = np.mean(np.abs(E_error_lf), axis=1)  # [steps]
        avg_E_error_rk4 = np.mean(np.abs(E_error_rk4), axis=1)  # [steps]

        plt.figure(figsize=(12, 8))
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
        plt.close()
        logging.info(f"Energy conservation error plot saved to '{self.results_dir}/energy_error.png'.")

    def plot_angular_momentum_error(self):
        """
        Plot the angular momentum conservation error over time.
        """
        logging.info("Generating angular momentum conservation error plot.")

        # Time array in physical units
        times_physical = self.times * self.time_scale  # Time in Myr

        # Compute angular momentum errors
        L0_lf = self.angular_momenta_lf[0]  # [N]
        L_error_lf = (self.angular_momenta_lf - L0_lf) / np.abs(L0_lf)  # [steps, N]

        L0_rk4 = self.angular_momenta_rk4[0]  # [N]
        L_error_rk4 = (self.angular_momenta_rk4 - L0_rk4) / np.abs(L0_rk4)  # [steps, N]

        # Compute average angular momentum error across all stars
        avg_L_error_lf = np.mean(np.abs(L_error_lf), axis=1)  # [steps]
        avg_L_error_rk4 = np.mean(np.abs(L_error_rk4), axis=1)  # [steps]

        plt.figure(figsize=(12, 8))
        plt.plot(times_physical, avg_L_error_lf, label='Leapfrog', linewidth=1)
        plt.plot(times_physical, avg_L_error_rk4, label='RK4', linestyle='--', linewidth=1)
        plt.xlabel('Time (Myr)', fontsize=14)
        plt.ylabel('Relative Angular Momentum Error', fontsize=14)
        plt.title('Angular Momentum Conservation Error Over Time', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'angular_momentum_error.png'))
        plt.close()
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
        plt.close()
        logging.info(f"Execution time comparison plot saved to '{self.results_dir}/execution_times.png'.")


def main():
    # ============================================================
    # Initial Conditions (Dimensionless Units)
    # ============================================================

    # Number of stars
    N_stars = 100  # You can adjust this number as needed

    # Maximum radial distance (Rmax) in dimensionless units
    Rmax = 10.0  # Adjust based on the simulation needs

    # Logging simulation properties
    logging.info("Starting the simulation with the following properties:")

    # Create Galaxy instance
    galaxy = Galaxy(mass=1.0, a=2.0, b=0.1)

    # Initialize stars with the Schwarzschild velocity distribution
    galaxy.initialize_stars(N=N_stars, Rmax=Rmax, alpha=0.05, q=0.6, max_iterations=100)

    # Create Simulation instance
    simulation = Simulation(galaxy=galaxy, dt=0.05, t_max=250.0)

    # Run the simulation
    simulation.run()

    # ============================================================
    # Generate Plots
    # ============================================================

    # Generate plots using the Simulation class methods
    simulation.plot_trajectories(subset=100)  # Plot a subset of 100 stars for clarity
    simulation.plot_energy_error()
    simulation.plot_angular_momentum_error()
    simulation.plot_execution_time()


if __name__ == '__main__':
    main()
