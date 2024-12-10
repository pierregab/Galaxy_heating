# integrators.py

from galaxy import Galaxy
import logging
import numpy as np

# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Integrator:
    """
    Integrator class containing integration methods.
    """

    def compute_total_energy(self, pos, vel, masses, pos_BH, vel_BH, perturbers_mass, galaxy, perturbers=None):
        """
        Compute the total energy of the system.

        Parameters:
            pos (np.ndarray): Positions of stars [N, 3].
            vel (np.ndarray): Velocities of stars [N, 3].
            masses (np.ndarray): Masses of stars [N].
            pos_BH (np.ndarray or None): Positions of perturbers [P, 3].
            vel_BH (np.ndarray or None): Velocities of perturbers [P, 3].
            perturbers_mass (np.ndarray or None): Masses of perturbers [P].
            galaxy (Galaxy): Galaxy instance containing the potential.
            perturbers (list or None): List of perturbers.

        Returns:
            float: Total energy of the system.
        """
        total_energy = 0.0

        # Stars' kinetic energy
        kinetic_energy_stars = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
        total_energy += kinetic_energy_stars

        # Stars' potential energy due to galaxy
        R_stars = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        z_stars = pos[:, 2]
        potential_energy_stars_galaxy = np.sum(galaxy.potential(R_stars, z_stars) * masses)
        total_energy += potential_energy_stars_galaxy

        # Stars' potential energy due to perturbers (with softening)
        if pos_BH is not None:
            for j in range(len(perturbers_mass)):
                delta_r = pos_BH[j] - pos  # [N, 3]
                r = np.linalg.norm(delta_r, axis=1)  # [N]
                softened_r = np.sqrt(r**2 + galaxy.epsilon**2)  # [N]
                potential_energy_stars_perturbers = -galaxy.G * perturbers_mass[j] * np.sum(masses / softened_r)
                total_energy += potential_energy_stars_perturbers

        # Perturbers' kinetic and potential energy
        if pos_BH is not None and vel_BH is not None:
            # Kinetic energy of perturbers
            kinetic_energy_perturbers = 0.5 * np.sum(perturbers_mass * np.sum(vel_BH**2, axis=1))
            total_energy += kinetic_energy_perturbers

            # Potential energy of perturbers due to galaxy
            R_BH = np.sqrt(pos_BH[:, 0]**2 + pos_BH[:, 1]**2)
            z_BH = pos_BH[:, 2]
            potential_energy_perturbers_galaxy = np.sum(galaxy.potential(R_BH, z_BH) * perturbers_mass)
            total_energy += potential_energy_perturbers_galaxy

            # Potential energy of perturbers due to other perturbers (with softening, unique pairs)
            for i in range(len(perturbers_mass)):
                for j in range(i + 1, len(perturbers_mass)):
                    delta_r = pos_BH[i] - pos_BH[j]
                    r = np.linalg.norm(delta_r)
                    softened_r = np.sqrt(r**2 + galaxy.epsilon**2)
                    potential_energy_perturbers = -galaxy.G * perturbers_mass[i] * perturbers_mass[j] / softened_r
                    total_energy += potential_energy_perturbers

        return total_energy

    def leapfrog(self, particles: list, galaxy: Galaxy, dt: float, steps: int) -> tuple:
        """
        Leapfrog integrator for orbit simulation, including the perturbers.

        Parameters:
            particles (list of Particle): List of particles to integrate.
            galaxy (Galaxy): The galaxy instance containing the potential and perturbers.
            dt (float): Time step.
            steps (int): Number of integration steps.

        Returns:
            tuple: Contains positions, velocities, energies, angular_momenta,
                energies_BH, positions_BH, velocities_BH, angular_momenta_BH,
                total_energy, energy_error
        """
        logging.info("Starting Leapfrog integration.")

        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N, 3))  # Updated to store full vector

        # Initialize total energy array
        total_energy = np.zeros(steps)
        # Initialize energy error array
        energy_error = np.zeros(steps)

        # Check if perturbers are present
        if hasattr(galaxy, 'perturbers') and len(galaxy.perturbers):
            P = len(galaxy.perturbers)
            energies_BH = np.zeros((P, steps))
            positions_BH = np.zeros((P, steps, 3))
            velocities_BH = np.zeros((P, steps, 3))
            angular_momenta_BH = np.zeros((steps, P, 3))  # Updated to store full vector

            # Initialize perturbers' positions, velocities, masses
            perturbers = galaxy.perturbers
            pos_BH = np.array([pert.position for pert in perturbers])   # [P, 3]
            vel_BH = np.array([pert.velocity for pert in perturbers])   # [P, 3]
            perturbers_mass = np.array([pert.M for pert in perturbers]) # [P]
        else:
            energies_BH = None
            positions_BH = None
            velocities_BH = None
            angular_momenta_BH = None
            pos_BH = None
            vel_BH = None
            perturbers_mass = None

        # Initialize positions and velocities for stars
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]
        masses = np.array([particle.mass for particle in particles])   # [N]

        # Compute initial accelerations
        if pos_BH is not None:
            # Acceleration due to galaxy and perturbers on stars
            acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N,3]
            
            # Compute acceleration on perturbers due to galaxy, stars, and other perturbers
            acc_BH_galaxy = galaxy.acceleration(pos_BH)  # [P,3]
            
            # Acceleration due to stars on perturbers
            delta_r_stars = pos - pos_BH[:, np.newaxis, :]  # [P,N,3]
            r_stars = np.linalg.norm(delta_r_stars, axis=2)  # [P,N]
            softened_r_stars = np.sqrt(r_stars**2 + galaxy.epsilon**2)  # [P,N]
            
            # Avoid division by zero
            softened_r_stars = np.where(softened_r_stars == 0, 1e-10, softened_r_stars)
            
            acc_BH_stars = galaxy.G * np.sum(
                masses[np.newaxis, :, np.newaxis] * delta_r_stars / softened_r_stars[:, :, np.newaxis]**3,
                axis=1
            )  # [P,3]
            
            # Acceleration due to other perturbers on perturbers
            if P > 1:
                delta_r_perturbers = pos_BH[:, np.newaxis, :] - pos_BH[np.newaxis, :, :]  # [P,P,3]
                r_perturbers = np.linalg.norm(delta_r_perturbers, axis=2)  # [P,P]
                softened_r_perturbers = np.sqrt(r_perturbers**2 + galaxy.epsilon**2)  # [P,P]
                
                # Avoid self-interaction by masking the diagonal
                mask = ~np.eye(P, dtype=bool)
                delta_r_perturbers = delta_r_perturbers[mask].reshape(P, P-1, 3)  # [P, P-1, 3]
                softened_r_perturbers = softened_r_perturbers[mask].reshape(P, P-1)  # [P, P-1]
                
                # Compute acceleration components from other perturbers
                acc_BH_perturbers = galaxy.G * np.sum(
                    perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers / softened_r_perturbers[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]
            else:
                acc_BH_perturbers = np.zeros_like(acc_BH_stars)  # [P,3]

            # Total acceleration on perturbers
            acc_BH = acc_BH_galaxy + acc_BH_stars + acc_BH_perturbers  # [P,3]
        else:
            acc = galaxy.acceleration(pos)  # [N,3]
            acc_BH = None

        # Update velocities by half-step
        vel_half = vel + 0.5 * dt * acc  # [N, 3]
        if vel_BH is not None:
            vel_BH_half = vel_BH + 0.5 * dt * acc_BH  # [P, 3]

        # Compute initial total energy
        initial_total_energy = self.compute_total_energy(
            pos, vel, masses, pos_BH, vel_BH, perturbers_mass, galaxy
        )
        initial_energy = initial_total_energy
        logging.info(f"Initial Total Energy: {initial_total_energy:.6e} (normalized units)")

        for step in range(steps):
            # --- Drift: Update positions ---
            pos += dt * vel_half  # [N, 3]
            positions[step] = pos

            if pos_BH is not None:
                pos_BH += dt * vel_BH_half  # [P, 3]
                positions_BH[:, step] = pos_BH

                # Update perturbers' positions
                for index, pert in enumerate(perturbers):
                    pert.position = pos_BH[index]

            # --- Compute new accelerations ---
            if pos_BH is not None:
                # Acceleration on stars due to galaxy and perturbers
                acc_new = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N,3]
                
                # Compute acceleration on perturbers due to galaxy, stars, and other perturbers
                acc_BH_galaxy_new = galaxy.acceleration(pos_BH)  # [P,3]
                
                # Acceleration due to stars on perturbers
                delta_r_stars_new = pos - pos_BH[:, np.newaxis, :]  # [P,N,3]
                r_stars_new = np.linalg.norm(delta_r_stars_new, axis=2)  # [P,N]
                softened_r_stars_new = np.sqrt(r_stars_new**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars_new = np.where(softened_r_stars_new == 0, 1e-10, softened_r_stars_new)
                acc_BH_stars_new = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars_new / softened_r_stars_new[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]
                
                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers_new = pos_BH[:, np.newaxis, :] - pos_BH[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers_new = np.linalg.norm(delta_r_perturbers_new, axis=2)  # [P,P]
                    softened_r_perturbers_new = np.sqrt(r_perturbers_new**2 + galaxy.epsilon**2)  # [P,P]
                    
                    # Avoid self-interaction by masking the diagonal
                    mask_new = ~np.eye(P, dtype=bool)
                    delta_r_perturbers_new = delta_r_perturbers_new[mask_new].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers_new = softened_r_perturbers_new[mask_new].reshape(P, P-1)  # [P, P-1]
                    
                    acc_BH_perturbers_new = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers_new / softened_r_perturbers_new[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc_BH_perturbers_new = np.zeros_like(acc_BH_stars_new)  # [P,3]

                # Total acceleration on perturbers
                acc_BH_new = acc_BH_galaxy_new + acc_BH_stars_new + acc_BH_perturbers_new  # [P,3]
            else:
                acc_new = galaxy.acceleration(pos)  # [N,3]
                acc_BH_new = None

            # --- Kick: Update velocities ---
            # Advance half-step velocities to next half-step
            vel_half += dt * acc_new  # [N, 3]
            # Now vel_half corresponds to v(t + 3dt/2), so to get v(t + dt):
            vel_full = vel_half - 0.5 * dt * acc_new  # [N, 3]
            velocities[step] = vel_full

            if pos_BH is not None:
                vel_BH_half += dt * acc_BH_new  # [P, 3]
                # Similarly, convert BH half-step vel to full-step vel
                vel_BH_full = vel_BH_half - 0.5 * dt * acc_BH_new  # [P, 3]
                velocities_BH[:, step] = vel_BH_full
                for index, pert in enumerate(perturbers):
                    pert.velocity = vel_BH_full[index]

            # --- Compute Energies and Angular Momenta for stars ---
            v_squared = np.sum(vel_full ** 2, axis=1)  # [N]
            kinetic_energy = 0.5 * masses * v_squared  # [N]

            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)   # [N]
            z = pos[:, 2]                              # [N]
            potential_energy = galaxy.potential(R, z) * masses  # [N]

            if pos_BH is not None:
                for j in range(len(perturbers)):
                    delta_r = pos_BH[j] - pos  # [N, 3]
                    r = np.linalg.norm(delta_r, axis=1)  # [N]
                    softened_r = np.sqrt(r**2 + galaxy.epsilon**2)  # [N]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        potential_energy_pert = -galaxy.G * perturbers_mass[j] * masses / softened_r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)
                    potential_energy += potential_energy_pert  # [N] 

            energies[step] = kinetic_energy + potential_energy  # [N]

            # Angular Momentum for stars
            angular_momenta[step] = masses[:, np.newaxis] * np.cross(pos, vel_full)  # [N, 3]

            # --- Compute Energies for Perturbers ---
            if pos_BH is not None and energies_BH is not None:
                for index, pert in enumerate(perturbers):
                    # Kinetic Energy of Perturber
                    KE_BH = 0.5 * perturbers_mass[index] * np.dot(vel_BH_full[index], vel_BH_full[index])

                    # Potential Energy due to Galaxy
                    R_BH = np.sqrt(pos_BH[index, 0]**2 + pos_BH[index, 1]**2)
                    z_BH = pos_BH[index, 2]
                    PE_BH_galaxy = pert.galaxy.potential(R_BH, z_BH)

                    # Potential Energy due to other perturbers
                    potential_energy_other_perturbers = 0.0
                    for j in range(len(perturbers)):
                        if j != index:
                            delta_r = pos_BH[index] - pos_BH[j]
                            r = np.linalg.norm(delta_r)
                            if r > 0:
                                potential_energy_other = -galaxy.G * perturbers_mass[index] * perturbers_mass[j] / r
                                potential_energy_other_perturbers += potential_energy_other

                    # Total Potential Energy of Perturber
                    PE_BH = perturbers_mass[index] * PE_BH_galaxy + potential_energy_other_perturbers

                    # Total Energy of Perturber
                    energies_BH[index, step] = KE_BH + PE_BH

                    # Angular Momentum for perturbers
                    angular_momenta_BH[step, index] = perturbers_mass[index] * np.cross(pos_BH[index], vel_BH_full[index])

            # --- Compute Total Energy ---
            current_total_energy = self.compute_total_energy(
                pos, vel_full, masses, pos_BH, vel_BH_full if pos_BH is not None else None,
                perturbers_mass, galaxy, perturbers if pos_BH is not None else None
            )
            total_energy[step] = current_total_energy

            # --- Compute Energy Error ---
            energy_error[step] = (current_total_energy - initial_energy) / np.abs(initial_energy)

            # --- Log progress and energy conservation ---
            if (step + 1) % max(1, steps // 10) == 0 or step == 0:
                progress_percent = 100 * (step + 1) / steps
                logging.info(f"Leapfrog integration progress: {progress_percent:.1f}%")
                logging.info(f"  Step {step + 1}/{steps}:")
                logging.info(f"    Total Energy: {current_total_energy:.6e}")
                logging.info(f"    Energy Error: {energy_error[step]:.6e} ({energy_error[step]*100:.4f}%)")

        logging.info("Leapfrog integration completed.")
        return (
            positions,
            velocities,
            energies,
            angular_momenta,
            energies_BH,
            positions_BH,
            velocities_BH,
            angular_momenta_BH,
            total_energy,
            energy_error
        )


    def rk4(self, particles: list, galaxy: Galaxy, dt: float, steps: int) -> tuple:
        """
        Runge-Kutta 4th order integrator for orbit simulation, including multiple perturbers.

        Parameters:
            particles (list of Particle): List of particles to integrate.
            galaxy (Galaxy): The galaxy instance containing the potential and perturbers.
            dt (float): Time step.
            steps (int): Number of integration steps.

        Returns:
            tuple: Contains positions, velocities, energies, angular_momenta,
                energies_BH, positions_BH, velocities_BH,
                angular_momenta_BH, total_energy, energy_error
        """
        logging.info("Starting RK4 integration.")

        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N, 3))  # Updated to store full vector

        # Initialize total energy array
        total_energy = np.zeros(steps)
        # Initialize energy error array
        energy_error = np.zeros(steps)

        # Check if perturbers are present
        if hasattr(galaxy, 'perturbers') and len(galaxy.perturbers) > 0:
            P = len(galaxy.perturbers)
            energies_BH = np.zeros((P, steps))
            positions_BH = np.zeros((P, steps, 3))
            velocities_BH = np.zeros((P, steps, 3))
            angular_momenta_BH = np.zeros((steps, P, 3))  # Updated to store full vector

            # Initialize perturbers' positions, velocities, masses
            perturbers = galaxy.perturbers
            pos_BH = np.array([pert.position for pert in perturbers])   # [P, 3]
            vel_BH = np.array([pert.velocity for pert in perturbers])   # [P, 3]
            perturbers_mass = np.array([pert.M for pert in perturbers]) # [P]
        else:
            energies_BH = None
            positions_BH = None
            velocities_BH = None
            angular_momenta_BH = None
            pos_BH = None
            vel_BH = None
            perturbers_mass = None

        # Initialize positions and velocities for stars
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]
        masses = np.array([particle.mass for particle in particles])   # [N]

        # Compute initial total energy
        initial_total_energy = self.compute_total_energy(
            pos, vel, masses, pos_BH, vel_BH, perturbers_mass, galaxy
        )
        initial_energy = initial_total_energy
        logging.info(f"Initial Total Energy: {initial_total_energy:.6e} (normalized units)")

        for step in range(steps):
            # --- k1 ---
            if pos_BH is not None:
                acc1 = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N, 3]
                # Compute acceleration on perturbers due to galaxy, stars, and other perturbers
                acc1_BH_galaxy = galaxy.acceleration(pos_BH)  # [P,3]

                # Acceleration due to stars on perturbers
                delta_r_stars = pos - pos_BH[:, np.newaxis, :]  # [P,N,3]
                r_stars = np.linalg.norm(delta_r_stars, axis=2)  # [P,N]
                softened_r_stars = np.sqrt(r_stars**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars = np.where(softened_r_stars == 0, 1e-10, softened_r_stars)
                acc1_BH_stars = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars / softened_r_stars[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]

                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers = pos_BH[:, np.newaxis, :] - pos_BH[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers = np.linalg.norm(delta_r_perturbers, axis=2)  # [P,P]
                    softened_r_perturbers = np.sqrt(r_perturbers**2 + galaxy.epsilon**2)  # [P,P]

                    # Avoid self-interaction by masking the diagonal
                    mask = ~np.eye(P, dtype=bool)
                    delta_r_perturbers = delta_r_perturbers[mask].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers = softened_r_perturbers[mask].reshape(P, P-1)  # [P, P-1]

                    # Compute acceleration components from other perturbers
                    acc1_BH_perturbers = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers / softened_r_perturbers[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc1_BH_perturbers = np.zeros_like(acc1_BH_stars)  # [P,3]

                # Total acceleration on perturbers
                acc1_BH = acc1_BH_galaxy + acc1_BH_stars + acc1_BH_perturbers  # [P,3]
            else:
                acc1 = galaxy.acceleration(pos)  # [N,3]
                acc1_BH = None

            # --- k2 ---
            pos_k2 = pos + 0.5 * dt * vel
            vel_k2 = vel + 0.5 * dt * acc1
            if pos_BH is not None:
                pos_BH_k2 = pos_BH + 0.5 * dt * vel_BH
                vel_BH_k2 = vel_BH + 0.5 * dt * acc1_BH

                acc2 = galaxy.acceleration(pos_k2, perturbers_pos=pos_BH_k2, perturbers_mass=perturbers_mass)  # [N,3]

                # Compute acceleration on perturbers at k2
                acc2_BH_galaxy = galaxy.acceleration(pos_BH_k2)  # [P,3]

                # Acceleration due to stars on perturbers
                delta_r_stars_k2 = pos_k2 - pos_BH_k2[:, np.newaxis, :]  # [P,N,3]
                r_stars_k2 = np.linalg.norm(delta_r_stars_k2, axis=2)  # [P,N]
                softened_r_stars_k2 = np.sqrt(r_stars_k2**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars_k2 = np.where(softened_r_stars_k2 == 0, 1e-10, softened_r_stars_k2)
                acc2_BH_stars = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars_k2 / softened_r_stars_k2[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]

                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers_k2 = pos_BH_k2[:, np.newaxis, :] - pos_BH_k2[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers_k2 = np.linalg.norm(delta_r_perturbers_k2, axis=2)  # [P,P]
                    softened_r_perturbers_k2 = np.sqrt(r_perturbers_k2**2 + galaxy.epsilon**2)  # [P,P]

                    # Avoid self-interaction by masking the diagonal
                    mask_k2 = ~np.eye(P, dtype=bool)
                    delta_r_perturbers_k2 = delta_r_perturbers_k2[mask_k2].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers_k2 = softened_r_perturbers_k2[mask_k2].reshape(P, P-1)  # [P, P-1]

                    # Compute acceleration components from other perturbers
                    acc2_BH_perturbers = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers_k2 / softened_r_perturbers_k2[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc2_BH_perturbers = np.zeros_like(acc2_BH_stars)  # [P,3]

                # Total acceleration on perturbers at k2
                acc2_BH = acc2_BH_galaxy + acc2_BH_stars + acc2_BH_perturbers  # [P,3]
            else:
                acc2 = galaxy.acceleration(pos_k2)  # [N,3]
                acc2_BH = None

            # --- k3 ---
            pos_k3 = pos + 0.5 * dt * vel_k2
            vel_k3 = vel + 0.5 * dt * acc2
            if pos_BH is not None:
                pos_BH_k3 = pos_BH + 0.5 * dt * vel_BH_k2
                vel_BH_k3 = vel_BH + 0.5 * dt * acc2_BH

                acc3 = galaxy.acceleration(pos_k3, perturbers_pos=pos_BH_k3, perturbers_mass=perturbers_mass)  # [N,3]

                # Compute acceleration on perturbers at k3
                acc3_BH_galaxy = galaxy.acceleration(pos_BH_k3)  # [P,3]

                # Acceleration due to stars on perturbers
                delta_r_stars_k3 = pos_k3 - pos_BH_k3[:, np.newaxis, :]  # [P,N,3]
                r_stars_k3 = np.linalg.norm(delta_r_stars_k3, axis=2)  # [P,N]
                softened_r_stars_k3 = np.sqrt(r_stars_k3**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars_k3 = np.where(softened_r_stars_k3 == 0, 1e-10, softened_r_stars_k3)
                acc3_BH_stars = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars_k3 / softened_r_stars_k3[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]

                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers_k3 = pos_BH_k3[:, np.newaxis, :] - pos_BH_k3[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers_k3 = np.linalg.norm(delta_r_perturbers_k3, axis=2)  # [P,P]
                    softened_r_perturbers_k3 = np.sqrt(r_perturbers_k3**2 + galaxy.epsilon**2)  # [P,P]

                    # Avoid self-interaction by masking the diagonal
                    mask_k3 = ~np.eye(P, dtype=bool)
                    delta_r_perturbers_k3 = delta_r_perturbers_k3[mask_k3].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers_k3 = softened_r_perturbers_k3[mask_k3].reshape(P, P-1)  # [P, P-1]

                    # Compute acceleration components from other perturbers
                    acc3_BH_perturbers = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers_k3 / softened_r_perturbers_k3[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc3_BH_perturbers = np.zeros_like(acc3_BH_stars)  # [P,3]

                # Total acceleration on perturbers at k3
                acc3_BH = acc3_BH_galaxy + acc3_BH_stars + acc3_BH_perturbers  # [P,3]
            else:
                acc3 = galaxy.acceleration(pos_k3)  # [N,3]
                acc3_BH = None

            # --- k4 ---
            pos_k4 = pos + dt * vel_k3
            vel_k4 = vel + dt * acc3
            if pos_BH is not None:
                pos_BH_k4 = pos_BH + dt * vel_BH_k3
                vel_BH_k4 = vel_BH + dt * acc3_BH

                acc4 = galaxy.acceleration(pos_k4, perturbers_pos=pos_BH_k4, perturbers_mass=perturbers_mass)  # [N,3]

                # Compute acceleration on perturbers at k4
                acc4_BH_galaxy = galaxy.acceleration(pos_BH_k4)  # [P,3]

                # Acceleration due to stars on perturbers
                delta_r_stars_k4 = pos_k4 - pos_BH_k4[:, np.newaxis, :]  # [P,N,3]
                r_stars_k4 = np.linalg.norm(delta_r_stars_k4, axis=2)  # [P,N]
                softened_r_stars_k4 = np.sqrt(r_stars_k4**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars_k4 = np.where(softened_r_stars_k4 == 0, 1e-10, softened_r_stars_k4)
                acc4_BH_stars = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars_k4 / softened_r_stars_k4[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]

                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers_k4 = pos_BH_k4[:, np.newaxis, :] - pos_BH_k4[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers_k4 = np.linalg.norm(delta_r_perturbers_k4, axis=2)  # [P,P]
                    softened_r_perturbers_k4 = np.sqrt(r_perturbers_k4**2 + galaxy.epsilon**2)  # [P,P]

                    # Avoid self-interaction by masking the diagonal
                    mask_k4 = ~np.eye(P, dtype=bool)
                    delta_r_perturbers_k4 = delta_r_perturbers_k4[mask_k4].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers_k4 = softened_r_perturbers_k4[mask_k4].reshape(P, P-1)  # [P, P-1]

                    # Compute acceleration components from other perturbers
                    acc4_BH_perturbers = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers_k4 / softened_r_perturbers_k4[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc4_BH_perturbers = np.zeros_like(acc4_BH_stars)  # [P,3]

                # Total acceleration on perturbers at k4
                acc4_BH = acc4_BH_galaxy + acc4_BH_stars + acc4_BH_perturbers  # [P,3]
            else:
                acc4 = galaxy.acceleration(pos_k4)  # [N,3]
                acc4_BH = None

            # --- Update positions and velocities ---
            pos += (dt / 6) * (vel + 2 * vel_k2 + 2 * vel_k3 + vel_k4)
            vel += (dt / 6) * (acc1 + 2 * acc2 + 2 * acc3 + acc4)

            if pos_BH is not None:
                pos_BH += (dt / 6) * (vel_BH + 2 * vel_BH_k2 + 2 * vel_BH_k3 + vel_BH_k4)
                vel_BH += (dt / 6) * (acc1_BH + 2 * acc2_BH + 2 * acc3_BH + acc4_BH)

                # Update perturbers' positions and velocities
                for index, pert in enumerate(perturbers):
                    pert.position = pos_BH[index]
                    pert.velocity = vel_BH[index]

            positions[step] = pos
            velocities[step] = vel

            if pos_BH is not None:
                positions_BH[:, step] = pos_BH
                velocities_BH[:, step] = vel_BH

            # --- Compute Energies and Angular Momenta for stars ---
            # Kinetic Energy for stars
            v_squared = np.sum(vel ** 2, axis=1)  # [N]
            kinetic_energy = 0.5 * masses * v_squared  # [N]

            # Potential Energy for stars due to galaxy
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)   # [N]
            z = pos[:, 2]                              # [N]
            potential_energy = galaxy.potential(R, z) * masses  # [N]

            # Potential Energy due to perturbers
            if pos_BH is not None:
                for j in range(len(perturbers)):
                    delta_r = pos_BH[j] - pos  # [N, 3]
                    r = np.linalg.norm(delta_r, axis=1)  # [N]
                    softened_r = np.sqrt(r**2 + galaxy.epsilon**2)  # [N]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        potential_energy_pert = -galaxy.G * perturbers_mass[j] * masses / softened_r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)
                    potential_energy += potential_energy_pert  # [N]

            # Total energy per particle
            energies[step] = kinetic_energy + potential_energy  # [N]

            # Angular Momentum for stars
            angular_momenta[step] = masses[:, np.newaxis] * np.cross(pos, vel)  # [N, 3]

            # --- Compute Energies and Angular Momenta for Perturbers ---
            if pos_BH is not None and energies_BH is not None:
                for index, pert in enumerate(perturbers):
                    # Kinetic Energy of Perturber
                    KE_BH = 0.5 * perturbers_mass[index] * np.dot(vel_BH[index], vel_BH[index])

                    # Potential Energy due to Galaxy
                    R_BH = np.sqrt(pos_BH[index, 0] ** 2 + pos_BH[index, 1] ** 2)
                    z_BH = pos_BH[index, 2]
                    PE_BH_galaxy = galaxy.potential(R_BH, z_BH) * perturbers_mass[index]

                    # Potential Energy due to other perturbers
                    potential_energy_other_perturbers = 0.0
                    for j in range(len(perturbers)):
                        if j != index:
                            delta_r = pos_BH[index] - pos_BH[j]
                            r = np.linalg.norm(delta_r)
                            softened_r = np.sqrt(r**2 + galaxy.epsilon**2)
                            if r > 0:
                                potential_energy_other = -galaxy.G * perturbers_mass[index] * perturbers_mass[j] / softened_r
                                potential_energy_other_perturbers += potential_energy_other

                    # Total Potential Energy of Perturber
                    PE_BH = PE_BH_galaxy + potential_energy_other_perturbers

                    # Total Energy of Perturber
                    energies_BH[index, step] = KE_BH + PE_BH

                    # Angular Momentum for Perturbers
                    angular_momenta_BH[step, index] = perturbers_mass[index] * np.cross(pos_BH[index], vel_BH[index])

            # --- Compute Total Energy ---
            current_total_energy = self.compute_total_energy(
                pos, vel, masses, pos_BH, vel_BH, perturbers_mass, galaxy, perturbers if pos_BH is not None else None
            )
            total_energy[step] = current_total_energy

            # --- Compute Energy Error ---
            energy_error[step] = (current_total_energy - initial_energy) / np.abs(initial_energy)

            # --- Log progress and energy conservation ---
            if (step + 1) % max(1, steps // 10) == 0 or step == 0:
                progress_percent = 100 * (step + 1) / steps
                logging.info(f"RK4 integration progress: {progress_percent:.1f}%")
                logging.info(f"  Step {step + 1}/{steps}:")
                logging.info(f"    Total Energy: {current_total_energy:.6e}")
                logging.info(f"    Energy Error: {energy_error[step]:.6e} ({energy_error[step]*100:.4f}%)")

        logging.info("RK4 integration completed.")
        return (
            positions,
            velocities,
            energies,
            angular_momenta,
            energies_BH,
            positions_BH,
            velocities_BH,
            angular_momenta_BH,
            total_energy,
            energy_error
        )


    def yoshida(self, particles: list, galaxy: Galaxy, dt: float, steps: int) -> tuple:
        """
        Yoshida 4th-order symplectic integrator for orbit simulation, including the perturbers.

        Parameters:
            particles (list of Particle): List of particles to integrate.
            galaxy (Galaxy): The galaxy instance containing the potential and perturbers.
            dt (float): Time step.
            steps (int): Number of integration steps.

        Returns:
            tuple: Contains positions, velocities, energies, angular_momenta,
                energies_BH, positions_BH, velocities_BH,
                angular_momenta_BH, total_energy, energy_error
        """
        logging.info("Starting Yoshida 4th-order symplectic integration.")

        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N, 3))  # Updated to store full vector

        # Initialize total energy array
        total_energy = np.zeros(steps)
        # Initialize energy error array
        energy_error = np.zeros(steps)

        # Coefficients for the Yoshida 4th-order symplectic integrator
        c1 = 0.6756035959798289
        c2 = -0.1756035959798288
        c3 = -0.1756035959798288
        c4 = 0.6756035959798289
        d1 = 1.3512071919596578
        d2 = -1.7024143839193153
        d3 = 1.3512071919596578

        # Check if perturbers are present
        if hasattr(galaxy, 'perturbers') and len(galaxy.perturbers):
            P = len(galaxy.perturbers)
            energies_BH = np.zeros((P, steps))
            positions_BH = np.zeros((P, steps, 3))
            velocities_BH = np.zeros((P, steps, 3))
            angular_momenta_BH = np.zeros((steps, P, 3))  # Updated to store full vector

            # Initialize perturbers' positions, velocities, masses
            perturbers = galaxy.perturbers
            pos_BH = np.array([pert.position for pert in perturbers])  # [P, 3]
            vel_BH = np.array([pert.velocity for pert in perturbers])  # [P, 3]
            perturbers_mass = np.array([pert.M for pert in perturbers])  # [P]
        else:
            energies_BH = None
            positions_BH = None
            velocities_BH = None
            angular_momenta_BH = None
            pos_BH = None
            vel_BH = None
            perturbers_mass = None

        # Initialize positions and velocities for stars
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]
        masses = np.array([particle.mass for particle in particles])   # [N]

        # Compute initial total energy
        initial_total_energy = self.compute_total_energy(
            pos, vel, masses, pos_BH, vel_BH, perturbers_mass, galaxy
        )
        initial_energy = initial_total_energy
        logging.info(f"Initial Total Energy: {initial_total_energy:.6e} (normalized units)")

        for step in range(steps):
            # --- First Substep ---
            # Drift
            pos += c1 * dt * vel
            if pos_BH is not None:
                pos_BH += c1 * dt * vel_BH
                # Update perturbers' positions
                for index, pert in enumerate(perturbers):
                    pert.position = pos_BH[index]

            # Kick
            if pos_BH is not None:
                acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N,3]

                # Compute acceleration on perturbers due to galaxy, stars, and other perturbers
                acc_BH_galaxy = galaxy.acceleration(pos_BH)  # [P,3]

                # Acceleration due to stars on perturbers
                delta_r_stars = pos - pos_BH[:, np.newaxis, :]  # [P,N,3]
                r_stars = np.linalg.norm(delta_r_stars, axis=2)  # [P,N]
                softened_r_stars = np.sqrt(r_stars**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars = np.where(softened_r_stars == 0, 1e-10, softened_r_stars)
                acc_BH_stars = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars / softened_r_stars[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]

                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers = pos_BH[:, np.newaxis, :] - pos_BH[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers = np.linalg.norm(delta_r_perturbers, axis=2)  # [P,P]
                    softened_r_perturbers = np.sqrt(r_perturbers**2 + galaxy.epsilon**2)  # [P,P]

                    # Avoid self-interaction by masking the diagonal
                    mask = ~np.eye(P, dtype=bool)
                    delta_r_perturbers = delta_r_perturbers[mask].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers = softened_r_perturbers[mask].reshape(P, P-1)  # [P, P-1]

                    # Compute acceleration components from other perturbers
                    acc_BH_perturbers = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers / softened_r_perturbers[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc_BH_perturbers = np.zeros_like(acc_BH_stars)  # [P,3]

                # Total acceleration on perturbers
                acc_BH = acc_BH_galaxy + acc_BH_stars + acc_BH_perturbers  # [P,3]
            else:
                acc = galaxy.acceleration(pos)  # [N,3]
                acc_BH = None

            vel += d1 * dt * acc
            if pos_BH is not None:
                vel_BH += d1 * dt * acc_BH
                # Update perturbers' velocities
                for index, pert in enumerate(perturbers):
                    pert.velocity = vel_BH[index]

            # --- Second Substep ---
            # Drift
            pos += c2 * dt * vel
            if pos_BH is not None:
                pos_BH += c2 * dt * vel_BH
                # Update perturbers' positions
                for index, pert in enumerate(perturbers):
                    pert.position = pos_BH[index]

            # Kick
            if pos_BH is not None:
                acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N,3]

                # Compute acceleration on perturbers
                acc_BH_galaxy = galaxy.acceleration(pos_BH)  # [P,3]

                # Acceleration due to stars on perturbers
                delta_r_stars_k2 = pos - pos_BH[:, np.newaxis, :]  # [P,N,3]
                r_stars_k2 = np.linalg.norm(delta_r_stars_k2, axis=2)  # [P,N]
                softened_r_stars_k2 = np.sqrt(r_stars_k2**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars_k2 = np.where(softened_r_stars_k2 == 0, 1e-10, softened_r_stars_k2)
                acc_BH_stars_k2 = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars_k2 / softened_r_stars_k2[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]

                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers_k2 = pos_BH[:, np.newaxis, :] - pos_BH[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers_k2 = np.linalg.norm(delta_r_perturbers_k2, axis=2)  # [P,P]
                    softened_r_perturbers_k2 = np.sqrt(r_perturbers_k2**2 + galaxy.epsilon**2)  # [P,P]

                    # Avoid self-interaction by masking the diagonal
                    mask_k2 = ~np.eye(P, dtype=bool)
                    delta_r_perturbers_k2 = delta_r_perturbers_k2[mask_k2].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers_k2 = softened_r_perturbers_k2[mask_k2].reshape(P, P-1)  # [P, P-1]

                    # Compute acceleration components from other perturbers
                    acc_BH_perturbers_k2 = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers_k2 / softened_r_perturbers_k2[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc_BH_perturbers_k2 = np.zeros_like(acc_BH_stars_k2)  # [P,3]

                # Total acceleration on perturbers at k2
                acc_BH_k2 = acc_BH_galaxy + acc_BH_stars_k2 + acc_BH_perturbers_k2  # [P,3]
            else:
                acc = galaxy.acceleration(pos)  # [N,3]
                acc_BH_k2 = None

            vel += d2 * dt * acc
            if pos_BH is not None:
                vel_BH += d2 * dt * acc_BH_k2
                # Update perturbers' velocities
                for index, pert in enumerate(perturbers):
                    pert.velocity = vel_BH[index]

            # --- Third Substep ---
            # Drift
            pos += c3 * dt * vel
            if pos_BH is not None:
                pos_BH += c3 * dt * vel_BH
                # Update perturbers' positions
                for index, pert in enumerate(perturbers):
                    pert.position = pos_BH[index]

            # Kick
            if pos_BH is not None:
                acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N,3]

                # Compute acceleration on perturbers
                acc_BH_galaxy = galaxy.acceleration(pos_BH)  # [P,3]

                # Acceleration due to stars on perturbers
                delta_r_stars_k3 = pos - pos_BH[:, np.newaxis, :]  # [P,N,3]
                r_stars_k3 = np.linalg.norm(delta_r_stars_k3, axis=2)  # [P,N]
                softened_r_stars_k3 = np.sqrt(r_stars_k3**2 + galaxy.epsilon**2)  # [P,N]
                softened_r_stars_k3 = np.where(softened_r_stars_k3 == 0, 1e-10, softened_r_stars_k3)
                acc_BH_stars_k3 = galaxy.G * np.sum(
                    masses[np.newaxis, :, np.newaxis] * delta_r_stars_k3 / softened_r_stars_k3[:, :, np.newaxis]**3,
                    axis=1
                )  # [P,3]

                # Acceleration due to other perturbers on perturbers
                if P > 1:
                    delta_r_perturbers_k3 = pos_BH[:, np.newaxis, :] - pos_BH[np.newaxis, :, :]  # [P,P,3]
                    r_perturbers_k3 = np.linalg.norm(delta_r_perturbers_k3, axis=2)  # [P,P]
                    softened_r_perturbers_k3 = np.sqrt(r_perturbers_k3**2 + galaxy.epsilon**2)  # [P,P]

                    # Avoid self-interaction by masking the diagonal
                    mask_k3 = ~np.eye(P, dtype=bool)
                    delta_r_perturbers_k3 = delta_r_perturbers_k3[mask_k3].reshape(P, P-1, 3)  # [P, P-1, 3]
                    softened_r_perturbers_k3 = softened_r_perturbers_k3[mask_k3].reshape(P, P-1)  # [P, P-1]

                    # Compute acceleration components from other perturbers
                    acc_BH_perturbers_k3 = galaxy.G * np.sum(
                        perturbers_mass[np.newaxis, 1:, np.newaxis] * delta_r_perturbers_k3 / softened_r_perturbers_k3[:, :, np.newaxis]**3,
                        axis=1
                    )  # [P,3]
                else:
                    acc_BH_perturbers_k3 = np.zeros_like(acc_BH_stars_k3)  # [P,3]

                # Total acceleration on perturbers at k3
                acc_BH_k3 = acc_BH_galaxy + acc_BH_stars_k3 + acc_BH_perturbers_k3  # [P,3]
            else:
                acc = galaxy.acceleration(pos)  # [N,3]
                acc_BH_k3 = None

            vel += d3 * dt * acc
            if pos_BH is not None:
                vel_BH += d3 * dt * acc_BH_k3
                # Update perturbers' velocities
                for index, pert in enumerate(perturbers):
                    pert.velocity = vel_BH[index]

            # --- Fourth Substep ---
            # Drift
            pos += c4 * dt * vel
            if pos_BH is not None:
                pos_BH += c4 * dt * vel_BH
                # Update perturbers' positions
                for index, pert in enumerate(perturbers):
                    pert.position = pos_BH[index]

            # --- Store positions and velocities ---
            positions[step] = pos
            velocities[step] = vel
            if pos_BH is not None:
                positions_BH[:, step] = pos_BH
                velocities_BH[:, step] = vel_BH

            # --- Compute Energies and Angular Momenta for stars ---
            # Kinetic Energy for stars
            v_squared = np.sum(vel ** 2, axis=1)  # [N]
            kinetic_energy = 0.5 * masses * v_squared  # [N]

            # Potential Energy for stars due to galaxy
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)   # [N]
            z = pos[:, 2]                              # [N]
            potential_energy = galaxy.potential(R, z) * masses  # [N]

            # Potential Energy due to perturbers
            if pos_BH is not None:
                for j in range(len(perturbers)):
                    delta_r = pos_BH[j] - pos  # [N, 3]
                    r = np.linalg.norm(delta_r, axis=1)  # [N]
                    softened_r = np.sqrt(r**2 + galaxy.epsilon**2)  # [N]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        potential_energy_pert = -galaxy.G * perturbers_mass[j] * masses / softened_r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)
                    potential_energy += potential_energy_pert  # [N]

            # Total energy per particle
            energies[step] = kinetic_energy + potential_energy  # [N]

            # Angular Momentum for stars
            angular_momenta[step] = masses[:, np.newaxis] * np.cross(pos, vel)  # [N, 3]

            # --- Compute Energies and Angular Momenta for Perturbers ---
            if pos_BH is not None and energies_BH is not None:
                for index, pert in enumerate(perturbers):
                    # Kinetic Energy of Perturber
                    KE_BH = 0.5 * perturbers_mass[index] * np.dot(vel_BH[index], vel_BH[index])

                    # Potential Energy due to Galaxy
                    R_BH = np.sqrt(pos_BH[index, 0] ** 2 + pos_BH[index, 1] ** 2)
                    z_BH = pos_BH[index, 2]
                    PE_BH_galaxy = galaxy.potential(R_BH, z_BH) * perturbers_mass[index]

                    # Potential Energy due to other perturbers
                    potential_energy_other_perturbers = 0.0
                    for j in range(len(perturbers)):
                        if j != index:
                            delta_r = pos_BH[index] - pos_BH[j]
                            r = np.linalg.norm(delta_r)
                            softened_r = np.sqrt(r**2 + galaxy.epsilon**2)
                            if r > 0:
                                potential_energy_other = -galaxy.G * perturbers_mass[index] * perturbers_mass[j] / softened_r
                                potential_energy_other_perturbers += potential_energy_other

                    # Total Potential Energy of Perturber
                    PE_BH = PE_BH_galaxy + potential_energy_other_perturbers

                    # Total Energy of Perturber
                    energies_BH[index, step] = KE_BH + PE_BH

                    # Angular Momentum for Perturbers
                    angular_momenta_BH[step, index] = perturbers_mass[index] * np.cross(pos_BH[index], vel_BH[index])

            # --- Compute Total Energy ---
            current_total_energy = self.compute_total_energy(
                pos, vel, masses, pos_BH, vel_BH, perturbers_mass, galaxy, perturbers if pos_BH is not None else None
            )
            total_energy[step] = current_total_energy

            # --- Compute Energy Error ---
            energy_error[step] = (current_total_energy - initial_energy) / np.abs(initial_energy)

            # --- Log progress and energy conservation ---
            if (step + 1) % max(1, steps // 10) == 0 or step == 0:
                progress_percent = 100 * (step + 1) / steps
                logging.info(f"Yoshida integration progress: {progress_percent:.1f}%")
                logging.info(f"  Step {step + 1}/{steps}:")
                logging.info(f"    Total Energy: {current_total_energy:.6e}")
                logging.info(f"    Energy Error: {energy_error[step]:.6e} ({energy_error[step]*100:.4f}%)")

        logging.info("Yoshida integration completed.")
        return (
            positions,
            velocities,
            energies,
            angular_momenta,
            energies_BH,
            positions_BH,
            velocities_BH,
            angular_momenta_BH,
            total_energy,
            energy_error
        )
