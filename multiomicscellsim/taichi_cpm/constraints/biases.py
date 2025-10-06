import taichi as ti
from multiomicscellsim.taichi_cpm.constraints.base import Constraint

"""
Constraints that does not depend on an energy landscape defined by cell configuration, but just from direct cell properties and local environment.
Examples are invasion penalty (a cell tries to avoid invading other cells) and chemotaxis (a cell tries to move up the gradient of a chemoattractant field).
"""


@ti.data_oriented
class InvasionPenaltyBias(Constraint):
    """
        A cell tries to avoid invading other cells.
    """
    energy_term_name: str = "invasion_penalty"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)

    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        """
            Calculate the current invasion penalty energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        # Invasion penalty is not cached
        return 0.0

    @ti.func
    def calculate_energy_delta(self, s_i, s_j, t_i, t_j):
        # There is no delta for invasion penalty, it is a bias term
        src_cell_id = self.sim.grid[s_i, s_j].cell_id
        tgt_cell_id = self.sim.grid[t_i, t_j].cell_id
        result = 0.0
        
        if tgt_cell_id != 0 and tgt_cell_id != src_cell_id:
            ti.atomic_add(result, self.lambda_weight)

        return result

@ti.data_oriented
class ChemotaxisBias(Constraint):
    """
        A cell tries to move up the gradient of a chemoattractant field.
    """
    energy_term_name: str = "chemotaxis"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)

    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        """
            Calculate the current chemotaxis energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        # Chemotaxis is not cached
        return 0.0

    @ti.func
    def calculate_energy_delta(self, s_i, s_j, t_i, t_j):
        # Chemotaxis biases the energy term based on the gradient of the chemoattractant field
        return -self.lambda_weight * (self.sim.chemokine_grid[t_i, t_j] - self.sim.chemokine_grid[s_i, s_j])



@ti.data_oriented
class RepulsionBias(Constraint):
    """
        A cell tries to move away from other cells.
    """
    energy_term_name: str = "repulsion"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)

    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        """
            Calculate the current repulsion energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        # Repulsion is not cached
        return 0.0
    
    @ti.func
    def calculate_energy_delta(self, s_i, s_j, t_i, t_j):
        # Repulsion biases the energy term based on the distance to other cells

        copy_direction = (ti.Vector([t_i, t_j]) - ti.Vector([s_i, s_j])).normalized()
        
        desired_direction = ti.Vector([0.0, 0.0])
        power = 0.0

        source_type = self.sim.cells[self.sim.grid[s_i, s_j].cell_id].cell_type
        target_type = self.sim.cells[self.sim.grid[t_i, t_j].cell_id].cell_type

        if source_type != target_type:
            if source_type == 0:
                # Source is background, energy is governed by target
                closest_center = ti.Vector([float("inf"), float("inf")])
                center = self.sim.cells[self.sim.grid[t_i, t_j].cell_id].center
                for c in range(self.sim.n_cells[None]):
                    if c > 0 and c != self.sim.grid[t_i, t_j].cell_id:
                        dist = (self.sim.cells[c].center - center).norm()
                        if dist < (closest_center - center).norm():
                            closest_center = self.sim.cells[c].center
                desired_direction = -(closest_center - center).normalized()
                power = 1.0
            else:
                # Source is a cell, energy is governed by source
                closest_center = ti.Vector([float("inf"), float("inf")])
                center = self.sim.cells[self.sim.grid[s_i, s_j].cell_id].center
                for c in range(self.sim.n_cells[None]):
                    if c > 0 and c != self.sim.grid[s_i, s_j].cell_id:
                        dist = (self.sim.cells[c].center - center).norm()
                        if dist < (closest_center - center).norm():
                            closest_center = self.sim.cells[c].center
                desired_direction = -(closest_center - center).normalized()
                power = 1.0
        dot = 0.0
        if desired_direction.norm() > 1e-6:
            desired_dir_norm = desired_direction.normalized()
            dot += desired_dir_norm.dot(copy_direction)

        return -self.lambda_weight * dot*power

@ti.data_oriented
class PolarizationBias(Constraint):
    """
        A cell favors copies in the direction of its polarization vector (major axis).

        This version is implemented as a bias term, that is, the cell shape does not affect the energy landscape.
        However, the bias is proportional to the current volume of the cell, so larger cells have a stronger bias.
        
        The effect is that cells tend to elongate or crawl in the direction of their polarization vector, 
        but the effect does not take into account the actual shape of the cell.
    """

    energy_term_name: str = "polarization_bias"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)

    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        """
            Calculate the current polarization energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        # Polarization is not cached
        return 0.0
    
    @ti.func
    def calculate_polarization_energy(self, s_i: int, s_j: int, t_i: int, t_j: int, cell_id: int) -> float:
        """
            Calculate the polarization energy of a cell in a given direction for a given copy attempt
        """
        energy = 0.0
        if cell_id > 0: # Ignore medium
            direction = ti.Vector([float(t_i - s_i), float(t_j - s_j)]).normalized()
            inbound = self.sim.grid[s_i, s_j].cell_id != cell_id # Is the copy inbound or outbound for the cell?
            dot = self.sim.cells[cell_id].maj_axis.dot(direction)
            # When copy is outbound, we want to favor either direction of the major axis (dot close to 1 or -1)
            if inbound == 0:
                energy = -self.lambda_weight * (dot**2)
            else:
                # When copy is inbound, we want to penalize against either direction of the major axis (dot close to 1 or -1)
                # TODO: Try also -self.lambda_weight * (1.0 - dot**2)
                energy = -self.lambda_weight * (1.0 - dot**2)
        return energy

    @ti.func
    def calculate_energy_delta(self, s_i: int, s_j: int, t_i: int, t_j: int) -> float:
        bias = 0.0  # Default bias
        bias += self.calculate_polarization_energy(s_i, s_j, t_i, t_j, self.sim.grid[s_i, s_j].cell_id)
        bias += self.calculate_polarization_energy(s_i, s_j, t_i, t_j, self.sim.grid[t_i, t_j].cell_id)
        return bias
    
