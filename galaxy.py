from system import System
from perturber import Perturber
from particle import Particle
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Galaxy(System):
    """
    Galaxy class representing the galactic potential.

    Attributes:
        M (float): Mass of the galaxy.
        a (float): Radial scale length.
        b (float): Vertical scale length.
        epsilon (float): Softening length for force smoothing.
        particles (list): List of star particles in the galaxy.
        perturbers (list): List of perturber instances in the galaxy.
    """

    def __init__(self, mass: float = 1.0, a: float = 2.0, b: float = None, epsilon: float = 0.01) -> None:
        super().__init__(self.__class__.__name__, M=mass)
        self.a = a           # Radial scale length (normalized to R0)
        self.b = 1 / 20 * self.a if b is None else b  # Vertical scale length (normalized)
        self.epsilon = epsilon  # Softening length for force smoothing

        # Log galaxy parameters
        logging.info(f"Galaxy parameters:")
        logging.info(f"  Mass (M): {self.M} (normalized)")
        logging.info(f"  Radial scale length (a): {self.a} (dimensionless)")
        logging.info(f"  Vertical scale length (b): {self.b} (dimensionless)")
        logging.info(f"  Softening length (epsilon): {self.epsilon} (dimensionless)")

        # Initialize list to hold Particle instances
        self.particles = []
        # Initialize list to hold Perturber instances
        self.perturbers = []

    def potential(self, R: float | np.ndarray, z: float | np.ndarray) -> float | np.ndarray:
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

    def set_perturbers(self, *perturbers: Perturber) -> "Galaxy":
        """
        Set the perturbers for the galaxy.

        Parameters:
            perturbers (Perturber): All perturber objects.
        """
        for pert in perturbers:
            pert.setHostGalaxy(self)
            self.perturbers.append(pert)

        logging.info(f"Perturbers have been set in the galaxy: {self.perturbers}")
        return self

    def acceleration(self, pos: np.ndarray, perturbers_pos: np.ndarray = None, perturbers_mass: np.ndarray = None) -> np.ndarray:
        """
        Compute the acceleration vectors at given positions, including the effect of the perturbers if present.

        Parameters:
            pos : np.ndarray
                Array of positions [N, 3].
            perturbers_pos : np.ndarray
                Positions of the perturbers [P, 3], default is None.
            perturbers_mass : np.ndarray
                Masses of the perturbers [P], default is None.

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

        # Add acceleration due to perturbers with force smoothing
        if perturbers_pos is not None and perturbers_mass is not None:
            for i, pos_pert in enumerate(perturbers_pos):
                delta = pos_pert - pos       # [N, 3]
                r_squared = np.sum(delta**2, axis=1).reshape(-1, 1)  # [N, 1]

                # Apply softening length epsilon
                softening = self.epsilon
                softened_r_squared = r_squared + softening**2
                softened_r_cubed = softened_r_squared ** 1.5

                with np.errstate(divide='ignore', invalid='ignore'):
                    acc_pert = self.G * perturbers_mass[i] * delta / softened_r_cubed  # [N, 3]
                acc_pert = np.nan_to_num(acc_pert)
                acc += acc_pert

        return acc

    def acceleration_single(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the acceleration at a single position due to the galaxy's potential.

        Parameters:
            pos : np.ndarray
                Position vector [3].

        Returns:
            np.ndarray
                Acceleration vector [3].
        """
        x, y, z = pos
        R = np.sqrt(x**2 + y**2)
        z_term = np.sqrt(z**2 + self.b**2)
        denom = (R**2 + (self.a + z_term)**2)**1.5
        ax = -self.G * self.M * x / denom
        ay = -self.G * self.M * y / denom
        az = -self.G * self.M * (self.a + z_term) * z / (z_term * denom)
        acc = np.array([ax, ay, az])

        return acc

    def dPhidr(self, R:float|np.ndarray[float], z:float|np.ndarray[float]=0) -> float|np.ndarray[float]:
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

    def omega(self, R:float|np.ndarray[float]) -> float|np.ndarray[float]:
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

    def kappa(self, R:float|np.ndarray[float]) -> float|np.ndarray[float]:
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
        d2Phidr2 = (-self.G * self.M / (denom)**3) * (1 - 3 * R**2 / denom)
        kappa_squared = 4 * Omega**2 + R * d2Phidr2
        kappa_squared = np.maximum(kappa_squared, 0)  # Avoid negative values due to numerical errors
        return np.sqrt(kappa_squared)
    
    def rho(self, R:float|np.ndarray[float], z:float|np.ndarray[float]=0) -> float|np.ndarray[float]:
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


    def initialize_stars(self, N:int, Rmax:float, alpha:float=0.05, max_iterations:int=100) -> None:
        """
        Initialize N stars with positions and velocities drawn from the Schwarzschild velocity distribution function.

        Parameters:
            N (int): Number of stars to initialize.
            Rmax (float): Maximum radial distance (dimensionless).
            alpha (float): Small parameter for radial displacement (default: 0.05).
            max_iterations (int): Maximum number of velocity regeneration iterations (default: 100).
        """
        logging.info(f"Initializing {N} stars using the Schwarzschild velocity distribution function.")

        # Calculate mass per star
        mass_per_star = self.M / N  # Distribute galaxy mass equally among stars
        logging.info(f"Mass per star: {mass_per_star:.6e} (normalized units)")

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

        # Corrected Radial velocity dispersion sigma_R
        sigma_R_squared = 2/3 * (alpha**2) * R_c**2 * kappa_squared
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
            #v_phi_new = v_c[idx_unbound] + (1 + 2*np.random.randint(-1,1)) * gamma[idx_unbound] * v_R_new 
            v_phi_new = v_c[idx_unbound] + np.random.normal(0, sigma_R[idx_unbound])

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
            mass_per_star = self.M / N  # Recalculate mass per star if some stars are removed
            logging.info(f"Proceeding with {N} bound stars with updated mass per star: {mass_per_star:.6e}")
        else:
            logging.info(f"All {N} stars initialized as bound orbits.")

        # Convert velocities to Cartesian coordinates
        v_x = v_R * np.cos(phi) - v_phi * np.sin(phi)
        v_y = v_R * np.sin(phi) + v_phi * np.cos(phi)

        # Create Particle instances with assigned mass
        for i in range(N):
            position = np.array([x_pos[i], y_pos[i], z_pos[i]])  # [x, y, z]
            velocity = np.array([v_x[i], v_y[i], v_z[i]])        # [vx, vy, vz]
            particle = Particle(position, velocity, mass=mass_per_star)  # Assign finite mass
            self.particles.append(particle)

        # Store initial data for later comparison
        self.initial_R = R.copy()
        self.initial_phi = phi.copy()
        self.initial_positions = np.column_stack((x_pos, y_pos, z_pos))
        self.initial_velocities = np.column_stack((v_x, v_y, v_z))
        self.initial_v_R = v_R.copy()
        self.initial_v_z = v_z.copy()
        self.initial_v_phi = v_phi.copy()

        logging.info(f"Initialization complete with {N} particles, each with mass {mass_per_star:.6e}.")

    def circular_velocity(self, R:float|np.ndarray[float]) -> float|np.ndarray[float]:
        """
        Compute the circular velocity at radius R.
        """
        dPhi_dR = self.dPhidr(R)
        v_c = np.sqrt(-R * dPhi_dR)
        return v_c