# galaxy.py

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


    def initialize_stars(self, N:int, Rmax:float, alpha:float=0.05, max_iterations:int=100, use_schwarzschild:bool=True) -> None:
        """
        Initialize N stars with positions and velocities drawn from the Schwarzschild velocity distribution function
        or placing them on circular orbits.

        Parameters:
            N (int): Number of stars to initialize.
            Rmax (float): Maximum radial distance (dimensionless).
            alpha (float): Small parameter for radial displacement (default: 0.05).
            max_iterations (int): Maximum number of velocity regeneration iterations (default: 100).
            use_schwarzschild (bool): Whether to use the Schwarzschild distribution for velocities (default: True).
        """
        if use_schwarzschild:
            self._initialize_stars_schwarzschild(N, Rmax, alpha, max_iterations)
        else:
            self._initialize_stars_circular(N, Rmax, alpha)

    def _initialize_stars_schwarzschild(self, N: int, Rmax: float, alpha: float, max_iterations: int) -> None:
        """
        Initialize stars using the Schwarzschild distribution with corrected vertical dynamics.
        """
        logging.info(f"Initializing {N} stars with corrected vertical velocity dispersion.")

        # Mass per star
        mass_per_star = self.M / N

        # Generate positions
        R_c = np.random.uniform(0, Rmax, N)  # Reference radii
        phi = np.random.uniform(0, 2 * np.pi, N)  # Angular positions
        x = np.random.uniform(-alpha * R_c, alpha * R_c, N)  # Radial displacements
        R = np.abs(R_c + x)  # Actual radial positions

        # Vertical positions (Gaussian distribution with scale height h = b)
        z_pos = np.random.normal(loc=0, scale=self.b, size=N)

        # Cartesian coordinates
        x_pos = R * np.cos(phi)
        y_pos = R * np.sin(phi)

        # Circular velocity and frequencies
        v_c = np.sqrt(R * (-self.dPhidr(R)))
        kappa = self.kappa(R)
        sigma_R = np.sqrt((2/3) * (alpha**2) * R_c**2 * kappa**2)

        # Corrected vertical velocity dispersion
        sigma_z_squared = (self.G * self.M * self.b) / (
            (R_c**2 + (self.a + self.b)**2)**(3/2)
        )
        sigma_z = np.sqrt(sigma_z_squared)

        # Store initial dispersions
        self.initial_sigma_R = sigma_R.copy()
        self.initial_sigma_z = sigma_z.copy()
        self.R_c = R_c.copy()

        # Velocity initialization loop
        v_R = np.zeros(N)
        v_phi = np.zeros(N)
        v_z = np.zeros(N)
        unbound = np.ones(N, dtype=bool)
        epsilon = 1e-6  # Avoid division by zero

        iterations = 0
        while np.any(unbound) and iterations < max_iterations:
            idx_unbound = np.where(unbound)[0]
            if len(idx_unbound) == 0:
                break

            # Generate velocities for unbound stars
            v_R_new = np.random.normal(0, sigma_R[idx_unbound])
            v_z_new = np.random.normal(0, sigma_z[idx_unbound])
            v_phi_mean = v_c[idx_unbound] - (sigma_R[idx_unbound]**2) / (2 * (v_c[idx_unbound] + epsilon))
            v_phi_new = v_phi_mean + np.random.normal(0, sigma_R[idx_unbound] / np.sqrt(2))

            # Energy check
            L_z_new = R[idx_unbound] * v_phi_new
            kinetic = 0.5 * (v_R_new**2 + v_z_new**2)
            rotational = 0.5 * (L_z_new**2) / R[idx_unbound]**2
            potential = self.potential(R[idx_unbound], z_pos[idx_unbound])
            E_total = kinetic + rotational + potential
            escape_speed_sq = -2 * potential
            total_speed_sq = v_R_new**2 + v_phi_new**2 + v_z_new**2
            unbound_new = (E_total >= 0) | (total_speed_sq >= escape_speed_sq)

            # Update velocities for bound stars
            bound = idx_unbound[~unbound_new]
            v_R[bound] = v_R_new[~unbound_new]
            v_phi[bound] = v_phi_new[~unbound_new]
            v_z[bound] = v_z_new[~unbound_new]
            unbound[idx_unbound] = unbound_new

            iterations += 1

        # Handle unbound stars
        if iterations == max_iterations and np.any(unbound):
            idx_bound = np.where(~unbound)[0]
            x_pos = x_pos[idx_bound]
            y_pos = y_pos[idx_bound]
            z_pos = z_pos[idx_bound]
            v_R = v_R[idx_bound]
            v_phi = v_phi[idx_bound]
            v_z = v_z[idx_bound]
            N = len(idx_bound)
            mass_per_star = self.M / N

        # Convert to Cartesian velocities
        v_x = v_R * np.cos(phi) - v_phi * np.sin(phi)
        v_y = v_R * np.sin(phi) + v_phi * np.cos(phi)

        # Create particles
        for i in range(N):
            position = np.array([x_pos[i], y_pos[i], z_pos[i]])
            velocity = np.array([v_x[i], v_y[i], v_z[i]])
            self.particles.append(Particle(position, velocity, mass_per_star))

        logging.info(f"Initialized {N} stars with stable σ_z.")

    def _initialize_stars_circular(self, N:int, Rmax:float, alpha:float=0.05) -> None:
        """
        Initialize stars on circular orbits.

        Parameters:
            N (int): Number of stars to initialize.
            Rmax (float): Maximum radial distance (dimensionless).
            alpha (float): Small parameter for radial displacement (default: 0.05).
        """
        logging.info(f"Initializing {N} stars on circular orbits.")

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

        # Store initial dispersions for later comparison (vectorized of zero)
        self.initial_sigma_R = np.zeros(N)
        self.initial_sigma_z = np.zeros(N)
        self.R_c = R_c.copy()  # Store R_c for each star

        # Positions in Cartesian coordinates
        x_pos = R * np.cos(phi)
        y_pos = R * np.sin(phi)
        z_pos = np.zeros(N)  # All stars lie in the galactic plane

        # Compute circular velocity at R
        v_c = self.circular_velocity(R)  # Circular velocity at R

        # Set velocities for circular orbits
        v_R = np.zeros(N)
        v_phi = v_c.copy()
        v_z = np.zeros(N)

        # Convert velocities to Cartesian coordinates
        v_x = -v_phi * np.sin(phi)  # v_x = v_phi * (-sin(phi))
        v_y = v_phi * np.cos(phi)   # v_y = v_phi * cos(phi)

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

        logging.info(f"Initialization complete with {N} particles on circular orbits, each with mass {mass_per_star:.6e}.")


    def initialize_stars_specific(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray = None) -> None:
        """
        Initialize multiple stars with specified positions and velocities.

        Parameters:
            positions (np.ndarray): Array of position vectors [N, 3].
            velocities (np.ndarray): Array of velocity vectors [N, 3].
            masses (np.ndarray, optional): Array of masses [N]. If None, masses are set to galaxy's mass divided by number of stars.
        """
        N = positions.shape[0]
        
        if masses is None:
            masses = np.full(N, self.M / N)
        elif len(masses) != N:
            raise ValueError("Length of masses array must match number of positions and velocities.")

        for i in range(N):
            particle = Particle(positions[i], velocities[i], masses[i])
            self.particles.append(particle)
            logging.info(f"Initialized star {i+1}/{N} at position {positions[i]} with velocity {velocities[i]} and mass {masses[i]:.6e}.")

    def circular_velocity(self, R:float|np.ndarray[float]) -> float|np.ndarray[float]:
        """
        Compute the circular velocity at radius R.

        Parameters:
            R : float or np.ndarray
                Radial distance(s) (dimensionless).

        Returns:
            float or np.ndarray
                Circular velocity v_c at each R.
        """
        dPhi_dR = self.dPhidr(R)
        v_c = np.sqrt(-R * dPhi_dR)
        return v_c
