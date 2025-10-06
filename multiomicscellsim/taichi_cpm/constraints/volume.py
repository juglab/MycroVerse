import taichi as ti
from multiomicscellsim.taichi_cpm.constraints.base import Constraint

@ti.data_oriented
class VolumeConstraint(Constraint):
    """
        A cell tries to maintain a target volume.
    """
    energy_term_name: str = "volume"

    def __init__(self, sim, energy_index:int, lambda_weight: float = 1.0):
        super().__init__(sim, energy_index, lambda_weight)

    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        """
            Calculate the current volume energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        current_volume = self.sim.cells[c].current_volume
        preferred_volume = self.sim.cells[c].preferred_volume
        return self.lambda_weight * (current_volume - preferred_volume)**2

    @ti.func
    def calculate_copy_energy(self, s_i:int, s_j:int, t_i:int, t_j:int) -> float:
        """
            Calculate the volume energy of a pixel assuming it is copied from s to t.
        """
        src_cell_id = self.sim.grid[s_i, s_j].cell_id
        tgt_cell_id = self.sim.grid[t_i, t_j].cell_id
        total_energy = 0.0
        # Taichi does not support nested for...
        for c in range(self.sim.n_cells[None]):
            if c > 0:
                if c == src_cell_id or c == tgt_cell_id:
                    gain = 0.0
                    if src_cell_id == c:
                        # Current Volume gain one pixel
                        gain += 1.0
                    if tgt_cell_id == c:
                        # Current Volume loses one pixel
                        gain -= 1.0
                    total_energy += self.lambda_weight * (self.sim.cells[c].current_volume + gain - self.sim.cells[c].preferred_volume)**2
        return total_energy

    @ti.func
    def calculate_energy_delta(self, s_i, s_j, t_i, t_j):
        # Volume: Hvol after copy - Current Hvol (0 gain given by src==target)
        current_energy_source = self.sim.cells[self.sim.grid[s_i, s_j].cell_id].current_energy_terms[self.energy_index]
        current_energy_target = self.sim.cells[self.sim.grid[t_i, t_j].cell_id].current_energy_terms[self.energy_index]
        delta_perimeter_current = current_energy_source + current_energy_target
        delta_volume_after = self.calculate_copy_energy(s_i, s_j, t_i, t_j)
        return delta_volume_after - delta_perimeter_current
