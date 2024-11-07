import logging
import numpy as np
# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        masses = np.array([particle.mass for particle in particles])  # [N]

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
            # Corrected kinetic energy computation
            v_squared = np.sum(vel_full ** 2, axis=1)  # [N]
            kinetic_energy = 0.5 * masses * v_squared  # [N]

            # Potential energy from the galaxy's potential
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z)     # [N]
            potential_energy *= masses  # [N]

            # Add potential energy due to the perturber
            if hasattr(galaxy, 'perturber'):
                delta_r = pos_BH - pos  # [N, 3]
                r = np.linalg.norm(delta_r, axis=1)  # [N]
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Compute PE = -G * m_p * m_i / r
                    potential_energy_pert = -galaxy.G * galaxy.perturber.mass * masses / r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)  # Replace NaNs with zero
                potential_energy += potential_energy_pert  # [N]

            # Total energy per particle
            energies[i] = kinetic_energy + potential_energy  # [N]

            # --- Compute angular momentum (Lz component) ---
            Lz = pos[:, 0] * vel_full[:, 1] - pos[:, 1] * vel_full[:, 0]  # [N]
            angular_momenta[i] = Lz * masses  # [N]

            # --- Compute and Store Perturber's Energy ---
            if hasattr(galaxy, 'perturber') and energies_BH is not None:
                # Kinetic Energy of Perturber
                KE_BH = 0.5 * galaxy.perturber.mass * np.dot(vel_BH_full, vel_BH_full)

                # Potential Energy due to Galaxy
                R_BH = np.sqrt(pos_BH[0]**2 + pos_BH[1]**2)
                PE_BH_galaxy = galaxy.potential(R_BH, pos_BH[2])

                # --- Potential Energy due to Stars ---
                # Compute the gravitational potential at the perturber's position due to all stars
                delta_r_BH = pos_BH - pos  # [N, 3]
                r_BH = np.linalg.norm(delta_r_BH, axis=1)  # [N]
                with np.errstate(divide='ignore', invalid='ignore'):
                    potential_energy_stars = -galaxy.G * np.sum(masses / r_BH)
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
        masses = np.array([particle.mass for particle in particles])  # [N]

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
            v_squared = np.sum(vel ** 2, axis=1)  # [N]
            kinetic_energy = 0.5 * masses * v_squared  # [N]

            # Potential Energy
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z)  # [N]
            potential_energy *= masses  # [N]

            if hasattr(galaxy, 'perturber'):
                # Potential energy due to perturber
                delta_r = pos_BH - pos  # [N, 3]
                r = np.linalg.norm(delta_r, axis=1)  # [N]
                with np.errstate(divide='ignore', invalid='ignore'):
                    potential_energy_pert = -galaxy.G * galaxy.perturber.mass * masses / r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)  # Replace NaNs with zero
                potential_energy += potential_energy_pert  # [N]

            energies[i] = kinetic_energy + potential_energy  # [N]

            # --- Compute Angular Momentum (Lz) ---
            Lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]  # [N]
            angular_momenta[i] = Lz * masses  # [N]

            # --- Compute and Store Perturber's Energy ---
            if hasattr(galaxy, 'perturber'):
                # Kinetic Energy of Perturber
                KE_BH = 0.5 * galaxy.perturber.mass * np.dot(vel_BH, vel_BH)

                # Potential Energy due to Galaxy
                R_BH = np.sqrt(pos_BH[0]**2 + pos_BH[1]**2)
                PE_BH_galaxy = galaxy.potential(R_BH, pos_BH[2])

                # --- Potential Energy due to Stars ---
                # Compute the gravitational potential at the perturber's position due to all stars
                delta_r_BH = pos_BH - pos  # [N, 3]
                r_BH = np.linalg.norm(delta_r_BH, axis=1)  # [N]
                with np.errstate(divide='ignore', invalid='ignore'):
                    potential_energy_stars = -galaxy.G * np.sum(masses / r_BH)
                    potential_energy_stars = 0.0 if np.isnan(potential_energy_stars) else potential_energy_stars

                # Total Potential Energy of Perturber
                PE_BH = PE_BH_galaxy + potential_energy_stars

                # Total Energy of Perturber
                energies_BH[i] = KE_BH + galaxy.perturber.mass * PE_BH

            # --- Log progress every 10% ---
            if (i + 1) % max(1, (steps // 10)) == 0:
                logging.info(f"RK4 integration progress: {100 * (i+1) / steps:.1f}%")

        logging.info("RK4 integration completed.")

        if hasattr(galaxy, 'perturber'):
            return positions, velocities, energies, angular_momenta, energies_BH, positions_BH, velocities_BH
        else:
            return positions, velocities, energies, angular_momenta

