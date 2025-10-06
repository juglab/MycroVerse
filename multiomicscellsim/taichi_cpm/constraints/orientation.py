import taichi as ti
from multiomicscellsim.taichi_cpm.constraints.base import Constraint
from multiomicscellsim.taichi_cpm.utils import compute_eigenvectors_and_eigenvalues

@ti.data_oriented
class PolarizationConstraint(Constraint):
    """
        A cell tries to maintain a target orientation.
    """
    energy_term_name: str = "polarization"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)

    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        pass

    @ti.func
    def calculate_energy_delta(self, s_i: int, s_j: int, t_i:int, t_j: int) -> float:
        """
            Calculate the polarization energy of a pixel assuming it is copied from s to t.
            The energy is calculated as the difference between the preferred polarization and the current polarization
        """ 
        
        t_pos = ti.Vector([float(t_i), float(t_j)])
        total_energy = 0.0
        N = int(0)
        new_N = int(0)
        gain = int(0)

        # If the source and target are the same, we just compute the current polarization energy

        if s_i == t_i and s_j == t_j:
            # If source and target are the same, we just compute the current polarization energy
            for c in range(self.sim.n_cells[None]):
                if c > 0:
                    if (c == self.sim.grid[s_i, s_j].cell_id or c == self.sim.grid[t_i, t_j].cell_id):
                        N = self.sim.cells[c].current_volume
                        new_N = N
                        
                        alignment = abs(self.sim.cells[c].current_polarization.dot(self.sim.cells[c].preferred_polarization))

                        #print(f"Cell {c} New Polarization: {new_polarization_vector}, Preferred: {preferred_polarization}, Alignment: {alignment}")
                        # TODO: Polarization strength
                        # TODO: When called with same source and target, should return the current polarization energy
                        ti.atomic_add(total_energy, self.lambda_weight * (1 - alignment)**2)
                        #ti.atomic_add(total_energy, 100*self.lambda_weight * (1.0 - self.cells[c].current_anisotropy)**2)
        else:
             for c in range(self.sim.n_cells[None]):
                gain = 0
                new_N = 0
                # All cells that are not source or target will keep the same polarization so they are not considered
                if c > 0:
                    if (c == self.sim.grid[s_i, s_j].cell_id or c == self.sim.grid[t_i, t_j].cell_id):
                        N = self.sim.cells[c].current_volume
                        if (c == self.sim.grid[s_i, s_j].cell_id):
                            # Source is gaining a pixel
                            gain = int(1)
                        if (c == self.sim.grid[t_i, t_j].cell_id):
                            # Target is losing a pixel
                            gain = int(-1)

                        new_N = self.sim.cells[c].current_volume + gain
                        # To compute the polarization energy after copy, we need to compute to consider that a gain/loss of pixel changes:
                        # 1. The center of mass
                        new_center = ti.Vector([0.0, 0.0])
                        
                        if (c == self.sim.grid[s_i, s_j].cell_id):
                            new_center = (N * self.sim.cells[c].center + t_pos) / new_N
                        elif (c == self.sim.grid[t_i, t_j].cell_id):
                            new_center = (N * self.sim.cells[c].center - t_pos) / new_N
                        
                        # 2. The covariance matrix
                        # Instead of recomputing the covariance matrix, we can use the current one and adjust it based on the new center of mass
                        # I.e., we assume the old pixels are still there, but the new pixel changes the center of mass and the covariance matrix
                        new_covariance_matrix = (1 / new_N) * (N * self.sim.cells[c].covariance_matrix + gain*(new_center - t_pos).outer_product(new_center - t_pos))
                        # 3. The polarization vector (longest axis of the covariance matrix)
                        new_polarization_vector, _, new_anisotropy = self.sim.compute_polarization_and_anisotropy(new_covariance_matrix)
                        
                        alignment = abs(new_polarization_vector.dot(self.sim.cells[c].preferred_polarization))
                        #print(f"Cell {c} New Polarization: {new_polarization_vector}, Preferred: {preferred_polarization}, Alignment: {alignment}")
                        # TODO: Polarization strength
                        ti.atomic_add(total_energy, self.lambda_weight * (1.0 - alignment)**2)
                        #ti.atomic_add(total_energy, 100*self.lambda_weight * (1.0 - new_anisotropy)**2)
        return total_energy
    

@ti.data_oriented
class AnisotropyOrientationConstraint(Constraint):
    energy_term_name: str = "orientation_anisotropy"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0, lambda_anisotropy: float = 1.0, lambda_orientation: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)
        self.lambda_anisotropy = lambda_anisotropy
        self.lambda_orientation = lambda_orientation


    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        energy = 0.0
        if c > 0:
            dot = self.sim.cells[c].maj_axis.dot(self.sim.cells[c].preferred_major_axis.normalized())
            dot = ti.min(1.0, ti.max(-1.0, dot)) # Avoid numerical issues
            energy += self.lambda_weight * (self.lambda_anisotropy * (self.sim.cells[c].current_anisotropy - self.sim.cells[c].preferred_anisotropy) ** 2 + \
                                            self.lambda_orientation * (1.0 - dot*dot))
        return energy

    @ti.func
    def calculate_energy_delta(self, s_i: int, s_j: int, t_i:int, t_j: int) -> float:
        
        # Delta energy is the difference between the current energy and the energy after the copy for both cells involved
        current_source_energy = self.sim.cells[self.sim.grid[s_i, s_j].cell_id].current_energy_terms[self.energy_index]
        current_target_energy = self.sim.cells[self.sim.grid[t_i, t_j].cell_id].current_energy_terms[self.energy_index]
        current_energy = current_source_energy + current_target_energy
        after_energy = self.energy_after_copy(s_i, s_j, t_i, t_j, self.sim.grid[s_i, s_j].cell_id) + \
                        self.energy_after_copy(s_i, s_j, t_i, t_j, self.sim.grid[t_i, t_j].cell_id)
        return after_energy - current_energy


    @ti.func
    def energy_after_copy(self, s_i:int, s_j:int, t_i:int, t_j:int, cell_id:int) -> float:
        """
            Calculate the orientation and anisotropy energy of a pixel assuming it is copied from s to t, considering only the given cell_id.
        """
        this_cell = self.sim.cells[cell_id]
        total = 0.0
        new_center = ti.Vector([0.0, 0.0])

        if cell_id > 0: # background is neutral to orientation/anisotropy changes
            # We need to compute the new anisotropy and orientation of the cell after the copy
            N_before = this_cell.current_volume # current volume
            gain = 1 if (cell_id == self.sim.grid[s_i, s_j].cell_id) else -1 if (cell_id == self.sim.grid[t_i, t_j].cell_id) else 0 # gain/loss of pixel
            N_after = ti.max(1.0, N_before + gain) # new volume, at least 1 to avoid division by zero
            # Position of the pixel being copied
            t_pos = ti.Vector([float(t_i), float(t_j)])
            
            # Compute new center with gain/loss of a pixel
            old_center = this_cell.center
            # center update
            if gain == 1:
                new_center += (N_before * this_cell.center + t_pos) / N_after
            elif gain == -1:
                new_center += (N_before * this_cell.center - t_pos) / N_after
            else:
                new_center += this_cell.center
            # Compute a new covariance matrix with gain/loss of a pixel
            if gain != 0:
                old_center = this_cell.center
                diff = t_pos - old_center
                new_cov = (N_before * this_cell.covariance_matrix + gain * diff.outer_product(diff)) / N_after

                # Recalculate orientation and anisotropy energy for the new cell configuration

                emax, emin, eva_max, eva_min = compute_eigenvectors_and_eigenvalues(new_cov)
                A_after = (eva_max - eva_min) / (eva_max + eva_min + 1e-6)
                A_star = this_cell.preferred_anisotropy
                E = self.lambda_anisotropy * (A_after - A_star) ** 2

                ustar = this_cell.preferred_major_axis.normalized()
                align2 = ti.min(1.0, ti.max(-1.0, emax.dot(ustar))) ** 2
                E += self.lambda_orientation * (1.0 - align2)

                total += E

        return total