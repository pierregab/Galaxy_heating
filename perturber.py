import numpy as np
import logging
# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def setGalaxy(self, galaxy):
        self.galaxy = galaxy

    def acceleration(self, pos):
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
        galaxy = self.galaxy
        z_term = np.sqrt(z**2 + galaxy.b**2)
        denom = (R**2 + (galaxy.a + z_term)**2)**1.5
        ax = -galaxy.G * galaxy.M * x / denom
        ay = -galaxy.G * galaxy.M * y / denom
        az = -galaxy.G * galaxy.M * (galaxy.a + z_term) * z / (z_term * denom)
        acc = np.array([ax, ay, az])

        return acc

    def reset(self):
        """
        Reset the perturber to its initial conditions.
        """
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        logging.info("Perturber reset to initial conditions.")
