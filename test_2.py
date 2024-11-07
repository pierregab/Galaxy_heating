import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.stats import norm
import timeit
import os  # Import the os module for directory operations
import concurrent.futures  # Import concurrent.futures for parallel execution

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
        set_perturber(perturber): Set the perturber for the galaxy.
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
    
    def set_perturber(self, perturber):
        """
        Set the perturber for the galaxy.

        Parameters:
            perturber (Perturber): The perturber object.
        """
        self.perturber = perturber
        logging.info("Perturber has been set in the galaxy.")


    def acceleration(self, pos, perturber_pos=None):
        """
        Compute the acceleration vectors at given positions, including the effect of the perturber if present.

        Parameters:
            pos : np.ndarray
                Array of positions [N, 3].
            perturber_pos : np.ndarray
                Position of the perturber [3], default is None.

        Returns:
            np.ndarray
                Array of acceleration vectors [N, 3].
        """
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        R = np.sqrt(x**2 + y**2)
        z_term = np.sqrt(z**2 + self.b**2)
        denom = (R**2 + (self.a + z_term)**2)**1.5
        ax = -self.G * self.M * x / denom
        ay = -self.G * self.M * y / denom
        az = -self.G * self.M * (self.a + z_term) * z / (z_term * denom)
        acc = np.stack((ax, ay, az), axis=-1)

        # Add acceleration due to perturber
        if perturber_pos is not None:
            delta_r = perturber_pos - pos  # [N, 3]
            r = np.linalg.norm(delta_r, axis=1)  # [N]
            with np.errstate(divide='ignore', invalid='ignore'):
                acc_pert = -self.G * self.perturber.mass * delta_r / r[:, None]**3
                acc_pert = np.nan_to_num(acc_pert)
            acc += acc_pert

        return acc

    def acceleration_single(self, pos):
        """
        Compute the acceleration at a single position due to the galaxy's potential.

        Parameters:
            pos : np.ndarray
                Position array [3].

        Returns:
            np.ndarray
                Acceleration vector [3].
        """
        x = pos[0]
        y = pos[1]
        z = pos[2]
        R = np.sqrt(x**2 + y**2)
        z_term = np.sqrt(z**2 + self.b**2)
        denom = (R**2 + (self.a + z_term)**2)**1.5
        ax = -self.G * self.M * x / denom
        ay = -self.G * self.M * y / denom
        az = -self.G * self.M * (self.a + z_term) * z / (z_term * denom)
        acc = np.array([ax, ay, az])

        return acc

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
        d2Phidr2 = (-self.G * self.M / (denom)**3) * (1 - 3 * R**2 / denom)
        kappa_squared = 4 * Omega**2 + R * d2Phidr2
        kappa_squared = np.maximum(kappa_squared, 0)  # Avoid negative values due to numerical errors
        return np.sqrt(kappa_squared)
    
    def rho(self, R, z=0):
        """
        Compute the mass density rho at a given R and z for the Miyamoto-Nagai potential.

        Parameters:
            R : float or np.ndarray
                Radial distance(s) (dimensionless).
            z : float or np.ndarray
                Vertical distance(s) (dimensionless), default is 0.

        Returns:
            float or np.ndarray
                Mass density rho at the specified location.
        """
        a = self.a
        b = self.b
        M = self.M

        # Compute terms
        D = np.sqrt(z**2 + b**2)
        denom = (R**2 + (a + D)**2)**(2.5)
        numerator = a * R**2 + (a + 3 * D) * (a + D)**2
        rho = M * b**2 * numerator / (4 * np.pi * denom * D**3)
        return rho

    def initialize_stars(self, N, Rmax, alpha=0.05, max_iterations=100):
            """
            Initialize N stars with positions and velocities drawn from the Schwarzschild velocity distribution function.

            Parameters:
                N (int): Number of stars to initialize.
                Rmax (float): Maximum radial distance (dimensionless).
                alpha (float): Small parameter for radial displacement (default: 0.05).
                max_iterations (int): Maximum number of velocity regeneration iterations (default: 100).
            """
            logging.info(f"Initializing {N} stars using the Schwarzschild velocity distribution function.")

            # Generate positions
            R_c = np.random.uniform(0, Rmax, N)  # Reference radii
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
            sigma_R_squared = (alpha**2) * R_c**2 * kappa_squared
            sigma_R = np.sqrt(sigma_R_squared)

            # Compute mass density rho at R_c, z=0
            rho_midplane = self.rho(R_c, z=0)

            # Corrected Vertical velocity dispersion sigma_z
            sigma_z_squared = self.b**2 * self.G * rho_midplane
            sigma_z = np.sqrt(sigma_z_squared)

            # Store initial dispersions for later comparison
            self.initial_sigma_R = sigma_R.copy()
            self.initial_sigma_z = sigma_z.copy()
            self.R_c = R_c.copy()  # Store R_c for each star

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
                sigma_R = sigma_R[idx_bound]
                sigma_z = sigma_z[idx_bound]
                R_c = R_c[idx_bound]
                self.initial_sigma_R = sigma_R.copy()
                self.initial_sigma_z = sigma_z.copy()
                self.R_c = R_c.copy()
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

            # Store initial data for later comparison
            self.initial_R = R.copy()
            self.initial_phi = phi.copy()
            self.initial_positions = np.column_stack((x_pos, y_pos, z_pos))
            self.initial_velocities = np.column_stack((v_x, v_y, v_z))
            self.initial_v_R = v_R.copy()
            self.initial_v_z = v_z.copy()
            self.initial_v_phi = v_phi.copy()

            logging.info(f"Initialization complete with {N} particles.")


class Particle:
    """
    Particle class representing a particle in the simulation.

    Attributes:
        position (np.ndarray): Position vector [x, y, z].
        velocity (np.ndarray): Velocity vector [vx, vy, vz].
        energy (float): Energy of the particle.
        angular_momentum (float): Angular momentum of the particle.
        mass (float): Mass of the particle.
    """

    def __init__(self, position, velocity, mass=0.0):
        self.initial_position = np.copy(position)
        self.initial_velocity = np.copy(velocity)
        self.position = np.copy(position)
        self.velocity = np.copy(velocity)
        self.energy = None
        self.angular_momentum = None
        self.mass = mass  # New attribute for mass

    def reset(self):
        """
        Reset the particle to its initial conditions.
        """
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        self.energy = None
        self.angular_momentum = None
        self.mass = 0.0  # Reset mass to zero if needed


class Perturber:
    """
    Perturber class representing a massive object (e.g., a black hole) that moves in the galaxy.

    Attributes:
        mass (float): Mass of the perturber (normalized).
        position (np.ndarray): Position vector [x, y, z].
        velocity (np.ndarray): Velocity vector [vx, vy, vz].
    """

    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.initial_position = np.copy(position)
        self.initial_velocity = np.copy(velocity)
        self.position = np.copy(position)
        self.velocity = np.copy(velocity)
        logging.info(f"Perturber initialized with mass {self.mass}, position {self.position}, and velocity {self.velocity}.")

    def reset(self):
        """
        Reset the perturber to its initial conditions.
        """
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        logging.info("Perturber reset to initial conditions.")



class Integrator:
    """
    Integrator class containing integration methods.
    """

    def leapfrog(self, particles, galaxy, dt, steps):
        """
        Leapfrog integrator for orbit simulation, including the perturber.
        
        This implementation follows the Kick-Drift-Kick scheme:
        1. Kick: Update velocities by half-step using current accelerations.
        2. Drift: Update positions by full-step using updated velocities.
        3. Kick: Update velocities by another half-step using new accelerations.
        
        Parameters:
            particles (list of Particle): List of particles to integrate.
            galaxy (Galaxy): The galaxy instance containing the potential and perturber.
            dt (float): Time step.
            steps (int): Number of integration steps.
        
        Returns:
            positions (np.ndarray): Positions of particles at each step [steps, N, 3].
            velocities (np.ndarray): Velocities of particles at each step [steps, N, 3].
            energies (np.ndarray): Total energies of particles at each step [steps, N].
            angular_momenta (np.ndarray): Angular momenta (Lz) of particles at each step [steps, N].
            energies_BH (np.ndarray or None): Total energies of the perturber at each step [steps] or None.
            positions_BH (np.ndarray or None): Positions of the perturber at each step [steps, 3] or None.
            velocities_BH (np.ndarray or None): Velocities of the perturber at each step [steps, 3] or None.
        """
        logging.info("Starting Leapfrog integration.")

        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N))
        
        if hasattr(galaxy, 'perturber'):
            energies_BH = np.zeros(steps)  # Array to store perturber's energy
        else:
            energies_BH = None

        # Initialize positions and velocities
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]

        # If there is a perturber, initialize its position and velocity
        if hasattr(galaxy, 'perturber'):
            perturber = galaxy.perturber
            pos_BH = np.copy(perturber.position)  # [3]
            vel_BH = np.copy(perturber.velocity)  # [3]

            # Store the perturber's trajectory
            positions_BH = np.zeros((steps, 3))
            velocities_BH = np.zeros((steps, 3))

            # Calculate initial acceleration for the perturber
            acc_BH = galaxy.acceleration_single(pos_BH)

            # Initialize half-step velocity for the perturber
            vel_BH_half = vel_BH + 0.5 * dt * acc_BH
        else:
            positions_BH = None
            velocities_BH = None

        # Calculate initial accelerations for stars using the perturber's position
        if hasattr(galaxy, 'perturber'):
            acc = galaxy.acceleration(pos, perturber_pos=pos_BH)  # [N, 3]
        else:
            acc = galaxy.acceleration(pos)  # [N, 3]

        # Update velocities by half-step
        vel_half = vel + 0.5 * dt * acc  # [N, 3]

        for i in range(steps):
            # --- Drift: Update positions by full-step using half-step velocities ---
            pos += dt * vel_half  # [N, 3]
            positions[i] = pos

            # Update perturber's position if present
            if positions_BH is not None:
                pos_BH += dt * vel_BH_half  # [3]
                positions_BH[i] = pos_BH

                # Update galaxy's perturber position
                galaxy.perturber.position = pos_BH

            # --- Compute new accelerations based on updated positions ---
            if hasattr(galaxy, 'perturber'):
                # Compute accelerations using the updated perturber position
                acc_new = galaxy.acceleration(pos, perturber_pos=pos_BH)  # [N, 3]
                acc_BH_new = galaxy.acceleration_single(pos_BH)  # [3]
            else:
                acc_new = galaxy.acceleration(pos)  # [N, 3]

            # --- Kick: Update velocities by full-step using new accelerations ---
            vel_half += dt * acc_new  # [N, 3]
            vel_full = vel_half - 0.5 * dt * acc_new  # [N, 3] (Full-step velocities)
            velocities[i] = vel_full

            # Update perturber's velocity if present
            if positions_BH is not None:
                vel_BH_half += dt * acc_BH_new  # [3]
                vel_BH_full = vel_BH_half - 0.5 * dt * acc_BH_new  # [3]
                velocities_BH[i] = vel_BH_full
                galaxy.perturber.velocity = vel_BH_full

            # --- Compute kinetic and potential energy for stars ---
            v = np.linalg.norm(vel_full, axis=1)  # [N]
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z)     # [N]

            # Add potential energy due to the perturber
            if hasattr(galaxy, 'perturber'):
                delta_r = pos_BH - pos  # [N, 3]
                r = np.linalg.norm(delta_r, axis=1)  # [N]
                # Handle cases where r is zero to avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    potential_energy_pert = - galaxy.G * galaxy.perturber.mass / r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)  # Replace NaNs with zero
                potential_energy += potential_energy_pert

            kinetic_energy = 0.5 * v**2  # [N]
            energies[i] = kinetic_energy + potential_energy  # [N]

            # --- Compute angular momentum (Lz component) ---
            Lz = pos[:, 0] * vel_full[:, 1] - pos[:, 1] * vel_full[:, 0]  # [N]
            angular_momenta[i] = Lz

            # --- Compute and Store Perturber's Energy ---
            if hasattr(galaxy, 'perturber') and energies_BH is not None:
                # Kinetic Energy of Perturber
                KE_BH = 0.5 * galaxy.perturber.mass * np.dot(vel_BH_full, vel_BH_full)

                # Potential Energy due to Galaxy
                R_BH = np.sqrt(pos_BH[0]**2 + pos_BH[1]**2)
                PE_BH_galaxy = galaxy.potential(R_BH, pos_BH[2])

                # --- NEW PART: Potential Energy due to Stars ---
                # Compute the gravitational potential at the perturber's position due to all stars
                delta_r_BH = pos_BH - pos  # [N, 3]
                r_BH = np.linalg.norm(delta_r_BH, axis=1)  # [N]
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Retrieve masses of all stars (should be zero for test particles)
                    masses = np.array([particle.mass for particle in particles])  # [N]
                    potential_energy_stars = - galaxy.G * np.sum(masses / r_BH)
                    potential_energy_stars = 0.0 if np.isnan(potential_energy_stars) else potential_energy_stars


                # Total Potential Energy of Perturber
                PE_BH = PE_BH_galaxy + potential_energy_stars

                # Total Energy of Perturber
                energies_BH[i] = KE_BH + galaxy.perturber.mass * PE_BH

            # --- Log progress every 10% ---
            if (i+1) % max(1, (steps // 10)) == 0:
                logging.info(f"Leapfrog integration progress: {100 * (i+1) / steps:.1f}%")

        logging.info("Leapfrog integration completed.")

        # Return positions and velocities of stars and perturber, along with energies
        return positions, velocities, energies, angular_momenta, energies_BH, positions_BH, velocities_BH


    def rk4(self, particles, galaxy, dt, steps):
        """
        Runge-Kutta 4th order integrator for orbit simulation, including the perturber.
        """
        logging.info("Starting RK4 integration.")
        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N))
        energies_BH = np.zeros(steps)  # Array to store perturber's energy
        positions_BH = None
        velocities_BH = None

        # Initialize positions and velocities for stars
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]

        # If there is a perturber, initialize its position and velocity
        if hasattr(galaxy, 'perturber'):
            perturber = galaxy.perturber
            pos_BH = np.copy(perturber.position)  # [3]
            vel_BH = np.copy(perturber.velocity)  # [3]

            # Store the perturber's trajectory
            positions_BH = np.zeros((steps, 3))
            velocities_BH = np.zeros((steps, 3))

        for i in range(steps):

            # --- RK4 for Stars and Perturber ---

            # k1 for perturber
            acc1_BH = galaxy.acceleration_single(pos_BH)  # [3]
            k1_vel_BH = dt * acc1_BH  # [3]
            k1_pos_BH = dt * vel_BH  # [3]

            # k1 for stars
            acc1 = galaxy.acceleration(pos, perturber_pos=pos_BH)  # [N, 3]
            k1_vel = dt * acc1  # [N, 3]
            k1_pos = dt * vel  # [N, 3]

            # k2 for perturber
            pos_BH_k2 = pos_BH + 0.5 * k1_pos_BH
            vel_BH_k2 = vel_BH + 0.5 * k1_vel_BH
            acc2_BH = galaxy.acceleration_single(pos_BH_k2)  # [3]
            k2_vel_BH = dt * acc2_BH  # [3]
            k2_pos_BH = dt * vel_BH_k2  # [3]

            # k2 for stars
            pos_k2 = pos + 0.5 * k1_pos
            vel_k2 = vel + 0.5 * k1_vel
            acc2 = galaxy.acceleration(pos_k2, perturber_pos=pos_BH_k2)  # [N, 3]
            k2_vel = dt * acc2  # [N, 3]
            k2_pos = dt * vel_k2  # [N, 3]

            # k3 for perturber
            pos_BH_k3 = pos_BH + 0.5 * k2_pos_BH
            vel_BH_k3 = vel_BH + 0.5 * k2_vel_BH
            acc3_BH = galaxy.acceleration_single(pos_BH_k3)  # [3]
            k3_vel_BH = dt * acc3_BH  # [3]
            k3_pos_BH = dt * vel_BH_k3  # [3]

            # k3 for stars
            pos_k3 = pos + 0.5 * k2_pos
            vel_k3 = vel + 0.5 * k2_vel
            acc3 = galaxy.acceleration(pos_k3, perturber_pos=pos_BH_k3)  # [N, 3]
            k3_vel = dt * acc3  # [N, 3]
            k3_pos = dt * vel_k3  # [N, 3]

            # k4 for perturber
            pos_BH_k4 = pos_BH + k3_pos_BH
            vel_BH_k4 = vel_BH + k3_vel_BH
            acc4_BH = galaxy.acceleration_single(pos_BH_k4)  # [3]
            k4_vel_BH = dt * acc4_BH  # [3]
            k4_pos_BH = dt * vel_BH_k4  # [3]

            # k4 for stars
            pos_k4 = pos + k3_pos
            vel_k4 = vel + k3_vel
            acc4 = galaxy.acceleration(pos_k4, perturber_pos=pos_BH_k4)  # [N, 3]
            k4_vel = dt * acc4  # [N, 3]
            k4_pos = dt * vel_k4  # [N, 3]

            # Update positions and velocities for perturber
            pos_BH += (k1_pos_BH + 2 * k2_pos_BH + 2 * k3_pos_BH + k4_pos_BH) / 6
            vel_BH += (k1_vel_BH + 2 * k2_vel_BH + 2 * k3_vel_BH + k4_vel_BH) / 6

            # Update positions and velocities for stars
            pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
            vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6

            positions[i] = pos
            velocities[i] = vel

            positions_BH[i] = pos_BH
            velocities_BH[i] = vel_BH

            # Update galaxy's perturber position and velocity
            galaxy.perturber.position = pos_BH
            galaxy.perturber.velocity = vel_BH

            # --- Compute Energies and Angular Momenta as before ---
            # Kinetic Energy
            v = np.linalg.norm(vel, axis=1)  # [N]
            kinetic_energy = 0.5 * v**2  # [N]

            # Potential Energy
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z)  # [N]

            if hasattr(galaxy, 'perturber'):
                # Potential energy due to perturber
                delta_r = pos_BH - pos  # [N, 3]
                r = np.linalg.norm(delta_r, axis=1)  # [N]
                potential_energy_pert = - galaxy.G * galaxy.perturber.mass / r  # [N]
                potential_energy += potential_energy_pert

            energies[i] = kinetic_energy + potential_energy  # [N]

            # --- Compute Angular Momentum (Lz) ---
            Lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]  # [N]
            angular_momenta[i] = Lz

            # --- Compute and Store Perturber's Energy ---
            if hasattr(galaxy, 'perturber'):
                # Kinetic Energy of Perturber
                # Corrected computation with mass included
                KE_BH = 0.5 * galaxy.perturber.mass * np.dot(vel_BH, vel_BH)
                PE_BH = galaxy.perturber.mass * galaxy.potential(np.sqrt(pos_BH[0]**2 + pos_BH[1]**2), pos_BH[2])

                # Total Energy of Perturber
                energies_BH[i] = KE_BH + PE_BH

            # --- Log progress every 10% ---
            if (i + 1) % max(1, (steps // 10)) == 0:
                logging.info(f"RK4 integration progress: {100 * (i+1) / steps:.1f}%")

        logging.info("RK4 integration completed.")

        if hasattr(galaxy, 'perturber'):
            return positions, velocities, energies, angular_momenta, energies_BH, positions_BH, velocities_BH
        else:
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
        compute_velocity_dispersions(): Compute and compare velocity dispersions.
        plot_velocity_histograms(): Plot histograms of initial and final velocity distributions.
        log_integrator_differences(): Compute and log differences between integrators.
    """

    def __init__(self, galaxy, dt=0.05, t_max=250.0, integrators=['Leapfrog', 'RK4']):
        """
        Initialize the Simulation.

        Parameters:
            galaxy (Galaxy): The galaxy instance.
            dt (float): Time step (dimensionless).
            t_max (float): Total simulation time (dimensionless).
            integrators (list): List of integrators to run. Options: 'Leapfrog', 'RK4'.
        """
        super().__init__()
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
        self.execution_times = {}
        self.energies_BH = {}  # Dictionary to store perturber's energy for each integrator
        self.perturber_positions = {}
        self.perturber_velocities = {}

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

    def reset_system(self):
        """
        Reset all particles and the perturber to their initial conditions.
        """
        logging.info("Resetting system to initial conditions.")
        # Reset all particles
        for particle in self.galaxy.particles:
            particle.reset()
        
        # Reset the perturber if it exists
        if hasattr(self.galaxy, 'perturber'):
            self.galaxy.perturber.reset()

    def run(self):
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
                pos, vel, energy, Lz, energies_BH, pos_BH, vel_BH = run_leapfrog()
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
                if pos_BH is not None:
                    self.perturber_positions['Leapfrog'] = pos_BH
                    self.perturber_velocities['Leapfrog'] = vel_BH

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
                    self.perturber_positions['RK4'] = pos_BH
                    self.perturber_velocities['RK4'] = vel_BH

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
            if integrator_name in self.perturber_positions:
                pos_BH = self.perturber_positions[integrator_name]  # [steps, 3]
                # x-y plot
                plt.subplot(1, 2, 1)
                plt.plot(pos_BH[:, 0] * self.length_scale,
                        pos_BH[:, 1] * self.length_scale,
                        color='red', linewidth=2, label='Perturber')
                plt.legend()

                # x-z plot
                plt.subplot(1, 2, 2)
                plt.plot(pos_BH[:, 0] * self.length_scale,
                        pos_BH[:, 2] * self.length_scale,
                        color='red', linewidth=2, label='Perturber')
                plt.legend()

            plt.tight_layout()
            filename = f'orbit_{integrator_name.lower()}.png'
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()
            logging.info(f"{integrator_name} orbit trajectories plots saved to '{self.results_dir}/{filename}'.")


    def plot_energy_error(self):
        """
        Plot the total energy conservation error over time for each integrator.
        """
        logging.info("Generating total energy conservation error plot.")

        plt.figure(figsize=(12, 8))

        for integrator_name in self.integrators:
            # Ensure that energy data for the integrator exists
            if integrator_name not in self.energies or (hasattr(self.galaxy, 'perturber') and integrator_name not in self.energies_BH):
                logging.warning(f"Energy data for integrator '{integrator_name}' is incomplete. Skipping.")
                continue

            # Time array in physical units
            times_physical = self.times * self.time_scale  # Time in Myr

            # Stars' energies: [steps, N]
            E_stars = self.energies[integrator_name]  # [steps, N]

            # Sum stars' energies at each step to get total stars' energy
            total_E_stars = np.sum(E_stars, axis=1)  # [steps]

            if hasattr(self.galaxy, 'perturber'):
                # Perturber's energies: [steps]
                E_BH = self.energies_BH[integrator_name]  # [steps]

                # System's total energy at each step
                total_E_system = total_E_stars + E_BH  # [steps]
            else:
                # If no perturber, system's total energy is stars' total energy
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



    def plot_angular_momentum_error(self):
        """
        Plot the angular momentum conservation error over time.
        """
        logging.info("Generating angular momentum conservation error plot.")

        plt.figure(figsize=(12, 8))
        for integrator_name in self.integrators:
            # Time array in physical units
            times_physical = self.times * self.time_scale  # Time in Myr

            # Compute angular momentum errors
            L0 = self.angular_momenta[integrator_name][0]  # [N]
            L_error = (self.angular_momenta[integrator_name] - L0) / np.abs(L0)  # [steps, N]

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

    def compute_velocity_dispersions(self):
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



    def plot_velocity_histograms(self, subset=200):
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

    def log_integrator_differences(self):
        """
        Compute and log the differences between RK4 and Leapfrog integrators for stars and the perturber.
        """
        logging.info("Computing differences between RK4 and Leapfrog integrators.")

        # Check if both integrators were run
        if 'RK4' not in self.integrators or 'Leapfrog' not in self.integrators:
            logging.warning("Both RK4 and Leapfrog integrators must be selected to compute differences.")
            return

        # Retrieve positions and velocities from both integrators
        positions_RK4 = self.positions['RK4']
        velocities_RK4 = self.velocities['RK4']
        positions_Leapfrog = self.positions['Leapfrog']
        velocities_Leapfrog = self.velocities['Leapfrog']

        # Ensure that both integrators have the same number of steps and particles
        if positions_RK4.shape != positions_Leapfrog.shape or velocities_RK4.shape != velocities_Leapfrog.shape:
            logging.error("Integrator results have mismatched shapes. Cannot compute differences.")
            return

        # Compute differences at the final time step for stars
        final_positions_RK4 = positions_RK4[-1]  # Shape: [N, 3]
        final_positions_Leapfrog = positions_Leapfrog[-1]
        final_velocities_RK4 = velocities_RK4[-1]
        final_velocities_Leapfrog = velocities_Leapfrog[-1]

        # Compute position and velocity differences for stars
        position_diff = final_positions_RK4 - final_positions_Leapfrog
        velocity_diff = final_velocities_RK4 - final_velocities_Leapfrog

        # Compute RMS differences for stars
        position_distance = np.sqrt(np.mean(np.sum(position_diff**2, axis=1)))
        velocity_distance = np.sqrt(np.mean(np.sum(velocity_diff**2, axis=1)))

        logging.info(f"Average position difference between RK4 and Leapfrog for stars: {position_distance:.6e} (dimensionless units)")
        logging.info(f"Average velocity difference between RK4 and Leapfrog for stars: {velocity_distance:.6e} (dimensionless units)")

        # If perturber data is available, compute differences for the perturber
        if 'RK4' in self.perturber_positions and 'Leapfrog' in self.perturber_positions:
            positions_BH_RK4 = self.perturber_positions['RK4']
            velocities_BH_RK4 = self.perturber_velocities['RK4']
            positions_BH_Leapfrog = self.perturber_positions['Leapfrog']
            velocities_BH_Leapfrog = self.perturber_velocities['Leapfrog']

            # Compute differences at the last time step for the perturber
            final_position_BH_RK4 = positions_BH_RK4[-1]
            final_position_BH_Leapfrog = positions_BH_Leapfrog[-1]
            final_velocity_BH_RK4 = velocities_BH_RK4[-1]
            final_velocity_BH_Leapfrog = velocities_BH_Leapfrog[-1]

            # Compute position and velocity differences for the perturber
            position_diff_BH = final_position_BH_RK4 - final_position_BH_Leapfrog
            velocity_diff_BH = final_velocity_BH_RK4 - final_velocity_BH_Leapfrog

            # Compute Euclidean distances for the perturber
            position_distance_BH = np.sqrt(np.sum(position_diff_BH**2))
            velocity_distance_BH = np.sqrt(np.sum(velocity_diff_BH**2))

            logging.info(f"Position difference between RK4 and Leapfrog for perturber: {position_distance_BH:.6e} (dimensionless units)")
            logging.info(f"Velocity difference between RK4 and Leapfrog for perturber: {velocity_distance_BH:.6e} (dimensionless units)")
        else:
            logging.warning("Perturber data is not available for both integrators.")


def main():
    # ============================================================
    # Initial Conditions (Dimensionless Units)
    # ============================================================

    # Number of stars
    N_stars = 1000  # Increased number for better statistics

    # Maximum radial distance (Rmax) in dimensionless units
    Rmax = 10.0  # Adjust based on the simulation needs

    # Logging simulation properties
    logging.info("Starting the simulation with the following properties:")

    # Create Galaxy instance
    galaxy = Galaxy(mass=1.0, a=2.0, b=0.1)

    # Initialize stars with the Schwarzschild velocity distribution
    galaxy.initialize_stars(N=N_stars, Rmax=Rmax, alpha=0.05, max_iterations=100)

    # Create Perturber instance
    M_BH = 0.1  # Mass of the perturber (normalized)
    initial_position_BH = np.array([5.0, 0.0, 10.0])  # Initial position [x, y, z]
    initial_velocity_BH = np.array([0.0, 0.0, 0.0])  # Initial velocity [vx, vy, vz]

    perturber = Perturber(mass=M_BH, position=initial_position_BH, velocity=initial_velocity_BH)

    # Set the perturber in the galaxy
    galaxy.set_perturber(perturber)

    # Compute an approximate orbital period at R=Rmax
    Omega_max = galaxy.omega(Rmax)
    T_orbit = 2 * np.pi / Omega_max  # Time for one orbit at Rmax

    # Total simulation time should be at least one orbital period at Rmax
    t_max = T_orbit * 1  # Simulate for 1 orbital period at Rmax

    # Time step
    dt = 0.01  # Smaller time step for better accuracy

    # Select integrators to run: 'Leapfrog', 'RK4', or both
    selected_integrators = ['RK4','Leapfrog']  # Modify this list to select integrators

    # Create Simulation instance with selected integrators
    simulation = Simulation(galaxy=galaxy, dt=dt, t_max=t_max, integrators=selected_integrators)

    # Run the simulation
    simulation.run()

    # ============================================================
    # Generate Plots
    # ============================================================

    # Generate plots using the Simulation class methods
    simulation.plot_trajectories(subset=200)  # Plot a subset of 200 stars for clarity
    simulation.plot_energy_error()
    simulation.plot_angular_momentum_error()
    simulation.plot_execution_time()

    # Compute and compare velocity dispersions
    simulation.compute_velocity_dispersions()

    # Plot velocity histograms
    simulation.plot_velocity_histograms(subset=200)

    # Compute and log differences between integrators
    simulation.log_integrator_differences()

    # Save positions and velocities for each integrator
    for integrator_name in selected_integrators:
        np.save(os.path.join(simulation.results_dir, f'positions_{integrator_name.lower()}.npy'),
                simulation.positions[integrator_name])
        np.save(os.path.join(simulation.results_dir, f'velocities_{integrator_name.lower()}.npy'),
                simulation.velocities[integrator_name])

    logging.info("All simulation data and plots have been saved.")


if __name__ == '__main__':
    main()
