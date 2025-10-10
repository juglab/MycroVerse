
import taichi as ti
from typing import List
from abc import ABC, abstractmethod
from multiomicscellsim.taichi_cpm.dynamics import BaseParameterDynamics
from multiomicscellsim.taichi_cpm.entities import Cell, CellType

@ti.data_oriented
class BaseBehaviour(ABC):
    """
        Base class for a behaviour.
        A behaviour implement a cell mechanisms that affects preferred parameters to implement some
        action strategy for the cell.
    """
    name: str
    influences: List[str]
    dynamics: BaseParameterDynamics

    def __init__(self, sim, dynamics: BaseParameterDynamics):
        self.sim = sim
        self.dynamics = dynamics

    @abstractmethod
    def on_behaviour_update(self, cell_id: int):
        """
            Action to be performed at each behaviour update step.
            NOTICE: This function is called INSIDE a Taichi kernel, so it must be Taichi-compatible.
            Moreover, any change to the cell parameters MUST be done through the sim.cells[cell_id]
        """
        pass

class VolumeBehaviour(BaseBehaviour):
    """
        A behaviour that makes a cell grow or shrink over time.
    """
    influences: List[str] = ["preferred_volume"]

    def __init__(self, sim, dynamics: BaseParameterDynamics):
        super().__init__(sim, dynamics)

    @ti.func
    def on_behaviour_update(self, cell_id: int):
        cell = self.sim.cells[cell_id]

    
class EllipticPerimeter(BaseBehaviour):
    """
       The cell tries to set its preferred perimeter to that of an ellipse of the same area and anisotropy.
       The perimeter is approximated using Ramanujan's formula.
    """
    name: str = "approximate_ellipse"
    influences: List[str] = ["preferred_perimeter"]

    def __init__(self, sim, dynamics: BaseParameterDynamics):
        super().__init__(sim, dynamics)

    @ti.func
    def on_behaviour_update(self, cell_id: int):

        cell = self.sim.cells[cell_id]
        if cell.cell_id > 0 and cell.cell_type > 0:
            a = ti.sqrt(cell.max_eigenvalue)
            b = ti.sqrt(cell.min_eigenvalue)

            scaling_factor = ti.sqrt(cell.current_volume / (ti.math.pi * a * b))
            a *= scaling_factor
            b *= scaling_factor

            # Step 3: Ramanujan's perimeter approximation
            h = ((a - b)**2) / ((a + b)**2)
            self.sim.cells[cell_id].preferred_perimeter = 3 * ti.math.pi * (a + b) * (1 + (3 * h) / (10 + ti.sqrt(4 - 3 * h)))
        
class MitosisAgeVolumeBehaviour(BaseBehaviour):
    """
        A behaviour that makes a cell more likely to divide as it approaches its preferred volume and age threshold.
    """
    name: str = "mitosis_age_volume"
    influences: List[str] = ["mitosis_probability"]

    def __init__(self, sim, dynamics: BaseParameterDynamics, sigmoid_slope: float = 0.1, probability_scale: float = 1.0):
        super().__init__(sim, dynamics)
        self.sigmoid_slope = sigmoid_slope
        self.probability_scale = probability_scale

    @ti.func
    def on_behaviour_update(self, cell_id: int):
        cell = self.sim.cells[cell_id]
        if cell.cell_id > 0 and cell.cell_type > 0:
            k = self.sigmoid_slope
            mitosis_prob_volume = (1.0 / (1.0 + ti.exp(-k * (self.sim.cells[cell_id].current_volume - self.sim.cells[cell_id].preferred_volume))))
            mitosis_prob_age = (1.0 / (1.0 + ti.exp(-k * (self.sim.cells[cell_id].current_age - self.sim.cells[cell_id].mitosis_age_threshold))))
            self.sim.cells[cell_id].mitosis_probability = mitosis_prob_volume * mitosis_prob_age * self.probability_scale
        