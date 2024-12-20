# system.py

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

    def __init__(self, className:str, M:float=None, log:bool=False) -> None:
        self.setM(M)                 # Mass (normalized)
        if log:
            # Logging simulation properties
            logging.info(f"Initializing system {className} with the following properties:")
            # Log the scaling factors
            logging.info(f"Physical units:")
            logging.info(f"  Gravitational constant (G_physical): {self.G_physical} kpc^3 Msun^-1 Myr^-2")
            logging.info(f"  Mass unit (M0): {self.mass_scale} Msun")
            logging.info(f"  Length unit (R0): {self.length_scale} kpc")
            logging.info(f"  Time unit (T0): {self.time_scale:.3f} Myr")
            logging.info(f"  Velocity unit (V0): {self.velocity_scale_kms:.3f} km/s")
        else:
            logging.info(f"Initializing system {className}")

    # ------------------------------------------------------------
    #           Physical Units (Astrophysical Units)
    # ------------------------------------------------------------
    ## Scaling Factors for conversion between simulation units and physical units
    # - Mass Unit (M0): 1 x 10^11 solar masses (Msun)
    # - Length Unit (R0): 1 kiloparsecs (kpc)
    # - Time Unit (T0): Derived from R0 and M0 using G
    # - Velocity Unit (V0): Derived from R0 and T0

    G_physical = 4.498e-12                                            # G = 4.498 x 10^-12 (kpc^3 Msun^-1 Myr^-2)
    mass_scale = 1e11                                                 # Mass unit in solar masses (Msun)
    length_scale = 1                                                  # Length unit in kiloparsecs (kpc)
    time_scale = np.sqrt(length_scale**3 / (G_physical * mass_scale)) # Time unit in Myr
    velocity_scale = length_scale / time_scale                        # Velocity unit in kpc/Myr
    velocity_scale_kms = velocity_scale * 977.8                       # Velocity unit in km/s (1 kpc/Myr = 977.8 km/s)

    # ------------------------------------------------------------
    #           Simulation Units (Dimensionless Units)
    # ------------------------------------------------------------
    # - Gravitational Constant (G): Set to 1 for normalization
    # - Mass (M): Set to 1 to normalize mass
    # - Radial Scale Length (a): Set to 2.0 (dimensionless)
    # - Vertical Scale Length (b): Set to 0.1 (dimensionless)

    G = 1.0                                                           # Gravitational constant (normalized to G_physical)
    M = 1.0                                                           # Main mass              (normalized to mass_scale)
    a = 2.0                                                           # Radial scale length    (normalized to length_scale)
    b = 0.1                                                           # Vertical scale length  (normalized to length_scale)

    def setM(self, M_value:float) -> "System":
        self.M = M_value
        return self
    
    