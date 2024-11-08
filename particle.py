import numpy as np

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

    def __init__(self, position:np.ndarray, velocity:np.ndarray, mass:float=0.0) -> None:
        self.initial_position = np.copy(position)
        self.initial_velocity = np.copy(velocity)
        self.position = np.copy(position)
        self.velocity = np.copy(velocity)
        self.energy = None
        self.angular_momentum = None
        self.mass = mass  # New attribute for mass

    def reset(self) -> "Particle":
        """
        Reset the particle to its initial conditions.
        """
        self.position = np.copy(self.initial_position)
        self.velocity = np.copy(self.initial_velocity)
        self.energy = None
        self.angular_momentum = None
        self.mass = 0.0  # Reset mass to 0.0
        return self
