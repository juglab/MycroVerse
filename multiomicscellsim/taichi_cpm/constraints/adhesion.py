import taichi as ti
from multiomicscellsim.taichi_cpm.constraints.base import Constraint

@ti.data_oriented
class AdhesionConstraint(Constraint):
    """
        A cell tries to maintain a target adhesion with other cells and the medium.
    """
    energy_term_name: str = "adhesion"

    @ti.func
    def calc_adhesion(self, i:int, j:int, cell_id:int) -> float:
        """
            Calculate the local adhesion of a pixel considering its cell_id as the given value
            so it can be used to simulate copies.
        """
        energy = 0.0
        for i_offset in range(-1, 2):
            for j_offset in range(-1, 2):
                if not self.sim.is_out_of_bounds(i+i_offset, j+j_offset) and \
                   cell_id != self.sim.grid[i+i_offset, j+j_offset].cell_id:
                   if self.sim.grid[i+i_offset, j+j_offset].cell_id == 0:
                       ti.atomic_add(energy, self.sim.cell_types[self.sim.cells[cell_id].cell_type].j_adhesion_stroma)
                   else:
                       ti.atomic_add(energy, self.sim.cell_types[self.sim.cells[cell_id].cell_type].j_adhesion_other)
        return energy
    
    @ti.func
    def calculate_current_cell_energy(self, c:int) -> float:
        """
            Calculate the  current adhesion energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        return 0.0 # Adhesion Currently not cached

    @ti.func
    def calculate_energy_delta(self, s_i, s_j, t_i, t_j):
        # Adhesion delta is calculated locally by summing adhesion of source and target pixels
        adhesion_after = self.calc_adhesion(t_i, t_j, self.sim.grid[s_i, s_j].cell_id) + self.calc_adhesion(s_i, s_j, self.sim.grid[s_i, s_j].cell_id)
        adhesion_current = self.calc_adhesion(t_i, t_j, self.sim.grid[t_i, t_j].cell_id) + self.calc_adhesion(s_i, s_j, self.sim.grid[s_i, s_j].cell_id)
        return adhesion_after - adhesion_current
