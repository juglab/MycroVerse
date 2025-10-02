import taichi as ti
from multiomicscellsim.taichi_cpm.constraints.base import Constraint
from multiomicscellsim.taichi_cpm.utils import local_perimeter

@ti.data_oriented
class PerimeterConstraint(Constraint):
    """
        A cell tries to maintain a target perimeter.
    """
    energy_term_name: str = "perimeter"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)

    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        """
            Calculate the current perimeter energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        current_perimeter = self.sim.cells[c].current_perimeter
        preferred_perimeter = self.sim.cells[c].preferred_perimeter
        return self.lambda_weight * (current_perimeter - preferred_perimeter)**2
    
    @ti.func
    def calculate_copy_energy(self, s_i:int, s_j:int, t_i:int, t_j:int, t_value:int) -> float:
        """
            Calculate the perimeter energy of a pixel assuming it is copied from s to t.
            To calculate the current perimeter, pass the same i,j as s_i, s_j and t_i, t_j
        """
        t_value = self.sim.grid[t_i, t_j].cell_id
        total_energy = 0.0

        for c in range(self.sim.n_cells[None]):
            # All cells that are not source or target will keep the same perimeter so they are not considered

            if c > 0 and (c == t_value or c == self.sim.grid[s_i, s_j].cell_id):
                gain_perimeter = 0.0
                current_perimeter = self.sim.cells[c].current_perimeter

                # If the source and target are the same, we have no gain, otherwise...
                if s_i != t_i or s_j != t_j:
                    # We just consider the neighborhood of the target pixel (which is the only one that changes)
                    same_cell_neighbors = 8 - local_perimeter(self.sim, t_i, t_j, c)
                    if c == t_value:
                        gain_perimeter = -8 + 2*same_cell_neighbors
                    else:
                        gain_perimeter = 8 - 2*same_cell_neighbors

                ti.atomic_add(total_energy, self.lambda_weight * (current_perimeter + gain_perimeter - self.sim.cells[c].preferred_perimeter)**2)

        return total_energy

    @ti.func
    def calculate_energy_delta(self, s_i, s_j, t_i, t_j):
        current_energy_source = self.sim.cells[self.sim.grid[s_i, s_j].cell_id].current_energy_terms[self.energy_index]
        current_energy_target = self.sim.cells[self.sim.grid[t_i, t_j].cell_id].current_energy_terms[self.energy_index]
        delta_perimeter_current = current_energy_source + current_energy_target
        delta_perimeter_after = self.calculate_copy_energy(s_i, s_j, t_i, t_j, self.sim.grid[t_i, t_j].cell_id)
        return delta_perimeter_after - delta_perimeter_current
    
    def on_behaviour_update(self, *args, **kwargs):
        return super().on_behaviour_update(*args, **kwargs)