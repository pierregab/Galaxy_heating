from galaxy import Galaxy
import logging
import numpy as np
# Set up professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Integrator:
    """
    Integrator class containing integration methods.
    """

    
    def leapfrog(self, particles: list, galaxy: Galaxy, dt: float, steps: int) -> tuple:
        """
        Leapfrog integrator for orbit simulation, including the perturbers.

        Parameters:
            particles (list of Particle): List of particles to integrate.
            galaxy (Galaxy): The galaxy instance containing the potential and perturbers.
            dt (float): Time step.
            steps (int): Number of integration steps.

        Returns:
            positions (np.ndarray): Positions of particles at each step [steps, N, 3].
            velocities (np.ndarray): Velocities of particles at each step [steps, N, 3].
            energies (np.ndarray): Total energies of particles at each step [steps, N].
            angular_momenta (np.ndarray): Angular momenta (Lz) of particles at each step [steps, N].
            energies_BH (np.ndarray or None): Total energies of the perturbers at each step [P, steps] or None.
            positions_BH (np.ndarray or None): Positions of the perturbers at each step [P, steps, 3] or None.
            velocities_BH (np.ndarray or None): Velocities of the perturbers at each step [P, steps, 3] or None.
        """
        logging.info("Starting Leapfrog integration.")

        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N))

        # Check if perturbers are present
        if hasattr(galaxy, 'perturbers') and len(galaxy.perturbers):
            P = len(galaxy.perturbers)
            energies_BH = np.zeros((P, steps))
            positions_BH = np.zeros((P, steps, 3))
            velocities_BH = np.zeros((P, steps, 3))

            # Initialize perturbers' positions, velocities, masses
            perturbers = galaxy.perturbers
            pos_BH = np.array([pert.position for pert in perturbers])   # [P, 3]
            vel_BH = np.array([pert.velocity for pert in perturbers])   # [P, 3]
            perturbers_mass = np.array([pert.M for pert in perturbers]) # [P]
            angular_momenta_BH = np.zeros((steps, len(perturbers)))     # [N, P]
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
        # Compute accelerations for perturbers including the galaxy's influence
        if pos_BH is not None:
            # Acceleration due to galaxy
            acc_BH_galaxy = galaxy.acceleration(pos_BH)  # [P, 3]
            
            # Acceleration due to other perturbers
            acc_BH_perturbers = np.zeros_like(pos_BH)
            for j in range(P):  # Renamed loop variable from 'i' to 'j'
                delta_r = pos_BH[j] - np.delete(pos_BH, j, axis=0)  # [P-1, 3]
                r = np.linalg.norm(delta_r, axis=1).reshape(-1, 1)  # [P-1, 1]
                masses_other = np.delete(perturbers_mass, j).reshape(-1, 1)  # [P-1, 1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    acc = -galaxy.G * masses_other * delta_r / r**3  # [P-1, 3]
                acc = np.nan_to_num(acc).sum(axis=0)  # Sum over all other perturbers
                acc_BH_perturbers[j] = acc
            
            # Total acceleration
            acc_BH = acc_BH_galaxy + acc_BH_perturbers  # [P, 3]
        else:
            acc_BH = None

        # Compute accelerations for stars
        if pos_BH is not None:
            acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N, 3]
        else:
            acc = galaxy.acceleration(pos)  # [N, 3]

        # Update velocities by half-step
        vel_half = vel + 0.5 * dt * acc  # [N, 3]
        if vel_BH is not None:
            vel_BH_half = vel_BH + 0.5 * dt * acc_BH  # [P, 3]

        for step in range(steps):  # Renamed loop variable from 'i' to 'step'
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
                acc_new = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N, 3]
                acc_BH_new = np.array([
                    pert.acceleration(
                        pos_self=pos_BH[j],
                        perturbers_pos=pos_BH,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)  # Renamed loop variable to 'j'
                ])  # [P, 3]
            else:
                acc_new = galaxy.acceleration(pos)  # [N, 3]
                acc_BH_new = None

            # --- Kick: Update velocities ---
            # Advance half-step velocities to next half-step
            vel_half += dt * acc_new  # [N, 3]
            vel_full = vel_half + 0.5 * dt * acc_new  # Corrected velocity update
            velocities[step] = vel_full

            if vel_BH is not None:
                vel_BH_half += dt * acc_BH_new  # [P, 3]
                vel_BH_full = vel_BH_half - 0.5 * dt * acc_BH_new  # [P, 3]
                velocities_BH[:, step] = vel_BH_full

                # Update perturbers' velocities
                for index, pert in enumerate(perturbers):
                    pert.velocity = vel_BH_full[index]

            # --- Compute Energies and Angular Momenta ---
            # Kinetic Energy for stars
            v_squared = np.sum(vel_full ** 2, axis=1)  # [N]
            kinetic_energy = 0.5 * masses * v_squared  # [N]

            # Potential Energy for stars due to galaxy
            R = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)   # [N]
            z = pos[:, 2]                              # [N]
            potential_energy = galaxy.potential(R, z) * masses  # [N]

            # Potential Energy due to perturbers
            if pos_BH is not None:
                for j in range(len(perturbers)):  # Use 'j' to avoid shadowing
                    delta_r = pos_BH[j] - pos  # [N, 3]
                    r = np.linalg.norm(delta_r, axis=1)  # [N]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        potential_energy_pert = -galaxy.G * perturbers_mass[j] * masses / r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)
                    potential_energy += potential_energy_pert  # [N] 

            # Total energy per particle
            energies[step] = kinetic_energy + potential_energy  # [N]

            # Angular Momentum Lz
            Lz = pos[:, 0] * vel_full[:, 1] - pos[:, 1] * vel_full[:, 0]  # [N]
            angular_momenta[step] = Lz * masses  # [N]

            # Angular Momentum for perturbers
            if pos_BH is not None:
                Lz_BH = pos_BH[:, 0] * vel_BH_full[:, 1] - pos_BH[:, 1] * vel_BH_full[:, 0]  # [P]
                angular_momenta_BH[i] = Lz_BH * np.array([pert.M for pert in perturbers])

            # --- Compute Energies for Perturbers ---
            if pos_BH is not None and energies_BH is not None:
                for index, pert in enumerate(perturbers):
                    # Kinetic Energy of Perturber
                    KE_BH = 0.5 * perturbers_mass[index] * np.dot(vel_BH_full[index], vel_BH_full[index])

                    # Potential Energy due to Galaxy
                    R_BH = np.sqrt(pos_BH[index, 0]**2 + pos_BH[index, 1]**2)
                    z_BH = pos_BH[index, 2]
                    PE_BH_galaxy = galaxy.potential(R_BH, z_BH)  # Use 'galaxy' instead of 'pert.galaxy'

                    # Potential Energy due to other perturbers
                    potential_energy_other_perturbers = 0.0
                    for j in range(P):  # Use 'j' to avoid shadowing
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

            # --- Log progress every 10% ---
            if (step + 1) % max(1, steps // 10) == 0:
                logging.info(f"Leapfrog integration progress: {100 * (step + 1) / steps:.1f}%")
        logging.info("Leapfrog integration completed.")
        return positions, velocities, energies, angular_momenta, energies_BH, positions_BH, velocities_BH, angular_momenta_BH
    

    def rk4(self, particles: list, galaxy: Galaxy, dt: float, steps: int) -> tuple:
        """
        Runge-Kutta 4th order integrator for orbit simulation, including multiple perturbers.

        Parameters:
            particles (list of Particle): List of particles to integrate.
            galaxy (Galaxy): The galaxy instance containing the potential and perturbers.
            dt (float): Time step.
            steps (int): Number of integration steps.

        Returns:
            positions (np.ndarray): Positions of particles at each step [steps, N, 3].
            velocities (np.ndarray): Velocities of particles at each step [steps, N, 3].
            energies (np.ndarray): Total energies of particles at each step [steps, N].
            angular_momenta (np.ndarray): Angular momenta (Lz) of particles at each step [steps, N].
            energies_BH (np.ndarray or None): Total energies of the perturbers at each step [P, steps] or None.
            positions_BH (np.ndarray or None): Positions of the perturbers at each step [P, steps, 3] or None.
            velocities_BH (np.ndarray or None): Velocities of the perturbers at each step [P, steps, 3] or None.
        """
        logging.info("Starting RK4 integration.")

        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N))

        # Check if perturbers are present
        if hasattr(galaxy, 'perturbers') and len(galaxy.perturbers) > 0:
            P = len(galaxy.perturbers)
            energies_BH = np.zeros((P, steps))
            positions_BH = np.zeros((P, steps, 3))
            velocities_BH = np.zeros((P, steps, 3))

            # Initialize perturbers' positions, velocities, masses
            perturbers = galaxy.perturbers
            pos_BH = np.array([pert.position for pert in perturbers])  # [P, 3]
            vel_BH = np.array([pert.velocity for pert in perturbers])  # [P, 3]
            perturbers_mass = np.array([pert.M for pert in perturbers])  # [P]
        else:
            energies_BH = None
            positions_BH = None
            velocities_BH = None
            pos_BH = None
            vel_BH = None
            perturbers_mass = None

        # Initialize positions and velocities for stars
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]
        masses = np.array([particle.mass for particle in particles])   # [N]

        for i in range(steps):

            # --- k1 ---
            if pos_BH is not None:
                acc1 = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)  # [N, 3]
                acc1_BH = np.array([
                    pert.acceleration(
                        pos_self=pos_BH[j],
                        perturbers_pos=pos_BH,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)
                ])  # [P, 3]
            else:
                acc1 = galaxy.acceleration(pos)  # [N, 3]
                acc1_BH = None

            k1_vel = dt * acc1  # [N, 3]
            k1_pos = dt * vel   # [N, 3]

            if pos_BH is not None:
                k1_vel_BH = dt * acc1_BH  # [P, 3]
                k1_pos_BH = dt * vel_BH   # [P, 3]

            # --- k2 ---
            pos_k2 = pos + 0.5 * k1_pos
            vel_k2 = vel + 0.5 * k1_vel
            if pos_BH is not None:
                pos_BH_k2 = pos_BH + 0.5 * k1_pos_BH
                vel_BH_k2 = vel_BH + 0.5 * k1_vel_BH

                acc2 = galaxy.acceleration(pos_k2, perturbers_pos=pos_BH_k2, perturbers_mass=perturbers_mass)
                acc2_BH = np.array([
                    pert.acceleration(
                        pos_self=pos_BH_k2[j],
                        perturbers_pos=pos_BH_k2,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)
                ])
                k2_vel_BH = dt * acc2_BH
                k2_pos_BH = dt * vel_BH_k2
            else:
                acc2 = galaxy.acceleration(pos_k2)
                acc2_BH = None
                k2_vel_BH = None
                k2_pos_BH = None

            k2_vel = dt * acc2
            k2_pos = dt * vel_k2

            # --- k3 ---
            pos_k3 = pos + 0.5 * k2_pos
            vel_k3 = vel + 0.5 * k2_vel
            if pos_BH is not None:
                pos_BH_k3 = pos_BH + 0.5 * k2_pos_BH
                vel_BH_k3 = vel_BH + 0.5 * k2_vel_BH

                acc3 = galaxy.acceleration(pos_k3, perturbers_pos=pos_BH_k3, perturbers_mass=perturbers_mass)
                acc3_BH = np.array([
                    pert.acceleration(
                        pos_self=pos_BH_k3[j],
                        perturbers_pos=pos_BH_k3,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)
                ])
                k3_vel_BH = dt * acc3_BH
                k3_pos_BH = dt * vel_BH_k3
            else:
                acc3 = galaxy.acceleration(pos_k3)
                acc3_BH = None
                k3_vel_BH = None
                k3_pos_BH = None

            k3_vel = dt * acc3
            k3_pos = dt * vel_k3

            # --- k4 ---
            pos_k4 = pos + k3_pos
            vel_k4 = vel + k3_vel
            if pos_BH is not None:
                pos_BH_k4 = pos_BH + k3_pos_BH
                vel_BH_k4 = vel_BH + k3_vel_BH

                acc4 = galaxy.acceleration(pos_k4, perturbers_pos=pos_BH_k4, perturbers_mass=perturbers_mass)
                acc4_BH = np.array([
                    pert.acceleration(
                        pos_self=pos_BH_k4[j],
                        perturbers_pos=pos_BH_k4,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)
                ])
                k4_vel_BH = dt * acc4_BH
                k4_pos_BH = dt * vel_BH_k4
            else:
                acc4 = galaxy.acceleration(pos_k4)
                acc4_BH = None
                k4_vel_BH = None
                k4_pos_BH = None

            k4_vel = dt * acc4
            k4_pos = dt * vel_k4

            # --- Update positions and velocities ---
            pos += (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos) / 6
            vel += (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel) / 6

            positions[i] = pos
            velocities[i] = vel

            if pos_BH is not None:
                pos_BH += (k1_pos_BH + 2 * k2_pos_BH + 2 * k3_pos_BH + k4_pos_BH) / 6
                vel_BH += (k1_vel_BH + 2 * k2_vel_BH + 2 * k3_vel_BH + k4_vel_BH) / 6

                positions_BH[:, i] = pos_BH
                velocities_BH[:, i] = vel_BH

                # Update perturbers' positions and velocities
                for index, pert in enumerate(perturbers):
                    pert.position = pos_BH[index]
                    pert.velocity = vel_BH[index]

            # --- Compute energies and angular momenta for stars ---
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
                    with np.errstate(divide='ignore', invalid='ignore'):
                        potential_energy_pert = -galaxy.G * perturbers_mass[j] * masses / r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)
                    potential_energy += potential_energy_pert  # [N]

            # Total energy per particle
            energies[i] = kinetic_energy + potential_energy  # [N]

            # Angular Momentum Lz
            Lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]  # [N]
            angular_momenta[i] = Lz * masses  # [N]

            # --- Compute Energies for Perturbers ---
            if pos_BH is not None and energies_BH is not None:
                for index, pert in enumerate(perturbers):
                    # Kinetic Energy of Perturber
                    KE_BH = 0.5 * pert.M * np.dot(vel_BH[index], vel_BH[index])

                    # Potential Energy due to Galaxy
                    R_BH = np.sqrt(pos_BH[index, 0]**2 + pos_BH[index, 1]**2)
                    z_BH = pos_BH[index, 2]
                    PE_BH_galaxy = pert.galaxy.potential(R_BH, z_BH)

                    # Potential Energy due to other perturbers
                    potential_energy_other_perturbers = 0.0
                    for j in range(index + 1, len(perturbers)):
                        delta_r = pos_BH[index] - pos_BH[j]
                        r = np.linalg.norm(delta_r)
                        if r > 0:
                            potential_energy_other = -galaxy.G * pert.M * perturbers_mass[j] / r
                            potential_energy_other_perturbers += potential_energy_other

                    # Total Potential Energy of Perturber
                    PE_BH = pert.M * PE_BH_galaxy + potential_energy_other_perturbers

                    # Total Energy of Perturber
                    energies_BH[index, i] = KE_BH + PE_BH

            # --- Log progress every 10% ---
            if (i + 1) % max(1, steps // 10) == 0:
                logging.info(f"RK4 integration progress: {100 * (i + 1) / steps:.1f}%")

        logging.info("RK4 integration completed.")
        return positions, velocities, energies, angular_momenta, energies_BH, positions_BH, velocities_BH
    
    def yoshida(self, particles: list, galaxy: Galaxy, dt: float, steps: int) -> tuple:
        """
        Yoshida 4th-order symplectic integrator for orbit simulation, including the perturbers.

        Parameters:
            particles (list of Particle): List of particles to integrate.
            galaxy (Galaxy): The galaxy instance containing the potential and perturbers.
            dt (float): Time step.
            steps (int): Number of integration steps.

        Returns:
            positions (np.ndarray): Positions of particles at each step [steps, N, 3].
            velocities (np.ndarray): Velocities of particles at each step [steps, N, 3].
            energies (np.ndarray): Total energies of particles at each step [steps, N].
            angular_momenta (np.ndarray): Angular momenta (Lz) of particles at each step [steps, N].
            energies_BH (np.ndarray or None): Total energies of the perturbers at each step [P, steps] or None.
            positions_BH (np.ndarray or None): Positions of the perturbers at each step [P, steps, 3] or None.
            velocities_BH (np.ndarray or None): Velocities of the perturbers at each step [P, steps, 3] or None.
        """
        logging.info("Starting Yoshida 4th-order symplectic integration.")

        N = len(particles)
        positions = np.zeros((steps, N, 3))
        velocities = np.zeros((steps, N, 3))
        energies = np.zeros((steps, N))
        angular_momenta = np.zeros((steps, N))

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

            # Initialize perturbers' positions, velocities, masses
            perturbers = galaxy.perturbers
            pos_BH = np.array([pert.position for pert in perturbers])  # [P, 3]
            vel_BH = np.array([pert.velocity for pert in perturbers])  # [P, 3]
            perturbers_mass = np.array([pert.M for pert in perturbers])  # [P]
        else:
            energies_BH = None
            positions_BH = None
            velocities_BH = None
            pos_BH = None
            vel_BH = None
            perturbers_mass = None

        # Initialize positions and velocities for stars
        pos = np.array([particle.position for particle in particles])  # [N, 3]
        vel = np.array([particle.velocity for particle in particles])  # [N, 3]
        masses = np.array([particle.mass for particle in particles])   # [N]

        for i in range(steps):
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
                acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)
                acc_BH = np.array([
                    pert.acceleration(
                        pos_self=pos_BH[j],
                        perturbers_pos=pos_BH,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)
                ])
            else:
                acc = galaxy.acceleration(pos)
                acc_BH = None

            vel += d1 * dt * acc
            if vel_BH is not None:
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
                acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)
                acc_BH = np.array([
                    pert.acceleration(
                        pos_self=pos_BH[j],
                        perturbers_pos=pos_BH,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)
                ])
            else:
                acc = galaxy.acceleration(pos)
                acc_BH = None

            vel += d2 * dt * acc
            if vel_BH is not None:
                vel_BH += d2 * dt * acc_BH
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
                acc = galaxy.acceleration(pos, perturbers_pos=pos_BH, perturbers_mass=perturbers_mass)
                acc_BH = np.array([
                    pert.acceleration(
                        pos_self=pos_BH[j],
                        perturbers_pos=pos_BH,
                        perturbers_mass=perturbers_mass
                    ) for j, pert in enumerate(perturbers)
                ])
            else:
                acc = galaxy.acceleration(pos)
                acc_BH = None

            vel += d3 * dt * acc
            if vel_BH is not None:
                vel_BH += d3 * dt * acc_BH
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

            # Store positions and velocities
            positions[i] = pos
            velocities[i] = vel
            if positions_BH is not None:
                positions_BH[:, i] = pos_BH
                velocities_BH[:, i] = vel_BH

            # --- Compute Energies and Angular Momenta ---
            # Kinetic Energy for stars
            v_squared = np.sum(vel ** 2, axis=1)  # [N]
            kinetic_energy = 0.5 * masses * v_squared  # [N]

            # Potential Energy for stars due to galaxy
            R = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)  # [N]
            z = pos[:, 2]  # [N]
            potential_energy = galaxy.potential(R, z) * masses  # [N]

            # Potential Energy due to perturbers
            if pos_BH is not None:
                for j in range(len(perturbers)):
                    delta_r = pos_BH[j] - pos  # [N, 3]
                    r = np.linalg.norm(delta_r, axis=1)  # [N]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        potential_energy_pert = -galaxy.G * perturbers_mass[j] * masses / r  # [N]
                    potential_energy_pert = np.nan_to_num(potential_energy_pert)
                    potential_energy += potential_energy_pert  # [N]

            # Total energy per particle
            energies[i] = kinetic_energy + potential_energy  # [N]

            # Angular Momentum Lz
            Lz = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]  # [N]
            angular_momenta[i] = Lz * masses  # [N]

            # --- Compute Energies for Perturbers ---
            if pos_BH is not None and energies_BH is not None:
                for index, pert in enumerate(perturbers):
                    # Kinetic Energy of Perturber
                    KE_BH = 0.5 * pert.M * np.dot(vel_BH[index], vel_BH[index])

                    # Potential Energy due to Galaxy
                    R_BH = np.sqrt(pos_BH[index, 0] ** 2 + pos_BH[index, 1] ** 2)
                    z_BH = pos_BH[index, 2]
                    PE_BH_galaxy = pert.galaxy.potential(R_BH, z_BH)

                    # Potential Energy due to other perturbers
                    potential_energy_other_perturbers = 0.0
                    for j in range(len(perturbers)):
                        if j != index:
                            delta_r = pos_BH[index] - pos_BH[j]
                            r = np.linalg.norm(delta_r)
                            if r > 0:
                                potential_energy_other = -galaxy.G * pert.M * perturbers_mass[j] / r
                                potential_energy_other_perturbers += potential_energy_other

                    # Total Potential Energy of Perturber
                    PE_BH = pert.M * PE_BH_galaxy + potential_energy_other_perturbers

                    # Total Energy of Perturber
                    energies_BH[index, i] = KE_BH + PE_BH

            # --- Log progress every 10% ---
            if (i + 1) % max(1, steps // 10) == 0:
                logging.info(f"Yoshida integration progress: {100 * (i + 1) / steps:.1f}%")

        logging.info("Yoshida integration completed.")
        return positions, velocities, energies, angular_momenta, energies_BH, positions_BH, velocities_BH, angular_momenta_BH

