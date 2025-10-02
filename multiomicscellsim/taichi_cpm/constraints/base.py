

import taichi as ti
from abc import ABC, abstractmethod

@ti.data_oriented
class Constraint(ABC):
    """
        Base class for constraints in the Cellular Potts Model (CPM) simulation.
        A simulation constraint is a term in the Hamiltonian that influences cell behavior.

        NOTICE: Keep in mind that constraints have read access to the entire simulation state.
        A constraint should not modify the simulation state directly.
        If a constraint needs to add a new energy term, it should be stored in the ce
    """
    lambda_weight: float
    energy_term_name: str = "base_constraint"
    energy_index: int = -1 # Energy index in the cell data array
    
    def __init__(self, sim, energy_index:int, lambda_weight: float, *args, **kwargs):
        self.sim = sim
        self.lambda_weight = lambda_weight
        self.energy_index = energy_index

    @abstractmethod
    def calculate_current_cell_energy(self, c, *args, **kwargs):
        """
            Calculate the current energy of a given cell.
            Used for storing the current energy state of a cell and optimizing performance.
        """
        pass

    @abstractmethod
    def calculate_energy_delta(self, s_i, s_j, t_i, t_j, *args, **kwargs):
        """
            Calculate the change in energy (delta H) for a proposed pixel copy attempt.
        """
        pass

    @abstractmethod
    def on_behaviour_update(self, *args, **kwargs):
        """
            Called at each behaviour update step to allow the constraint to update energy terms based on current simulation state.
        """
        pass




