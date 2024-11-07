import numpy as np
import logging
# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class System:
    '''
    System class : Base class for physical systems.
    @class.attribute
        - length_scale (float): Length scaling factor.
        - mass_scale (float): Mass scaling factor.
        - time_scale (float): Time scaling factor.
        - velocity_scale (float): Velocity scaling factor.
        - velocity_scale_kms (float): Velocity scaling factor in km/s.
        - G (float): Gravitational constant in simulation units (normalized).
    '''

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

