from system import System
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Perturber(System):
    """
    Perturber class representing a massive object (e.g., a black hole) that moves in the galaxy.

    Attributes:
        M (float): Mass of the perturber (normalized).
        position (np.ndarray): Position vector [x, y, z].
        velocity (np.ndarray): Velocity vector [vx, vy, vz].
    """

    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray) -> None:
        super().__init__(self.__class__.__name__, M=mass)
        self.initial_position = self.position = np.copy(position)
        self.initial_velocity = self.velocity = np.copy(velocity)
        logging.info(f"Perturber initialized with mass {self.M}, position {self.position}, and velocity {self.velocity}.")

    def setHostGalaxy(self, hostGalaxy) -> "Perturber":
        self.galaxy = hostGalaxy
        return self

    def acceleration(self, pos_self: np.ndarray, perturbers_pos: np.ndarray, perturbers_mass: np.ndarray) -> np.ndarray:
        """
        Compute the acceleration at the perturber's position due to the galaxy and other perturbers.

        Parameters:
            pos_self : np.ndarray
                Position of this perturber [3].
            perturbers_pos : np.ndarray
                Positions of all perturbers [P, 3].
            perturbers_mass : np.ndarray
                Masses of all perturbers [P].

        Returns:
            np.ndarray
                Acceleration vector [3].
        """
        # Acceleration due to the galaxy
        acc = self.galaxy.acceleration_single(pos_self)

        # Acceleration due to other perturbers
        for i, pos_other in enumerate(perturbers_pos):
            if np.array_equal(pos_self, pos_other):
                continue  # Skip self
            delta = pos_other - pos_self
            r = np.linalg.norm(delta)
            if r > 0:
                acc += self.G * perturbers_mass[i] * delta / r**3
        return acc

    def reset(self) -> "Perturber":
        """
        Reset the perturber to its initial conditions.
        """
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        logging.info("Perturber reset to initial conditions.")
        return self
