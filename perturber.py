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

    def reset(self):
        """
        Reset the perturber to its initial conditions.
        """
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        logging.info("Perturber reset to initial conditions.")
