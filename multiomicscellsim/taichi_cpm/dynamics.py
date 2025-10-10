
from abc import ABC, abstractmethod
from typing import List
from multiomicscellsim.taichi_cpm.entities import CellType

class BaseParameterDynamics(ABC):
    """
        Base class for parameter dynamics.
        A parameter dynamics implement a strategy to update preferred parameters of a cell over time.
    """
    kind: str
    def __init__(self):
        pass

    @abstractmethod
    def on_parameter_update(self, cell_type: CellType, step :int):
        pass

class ConstantParameterDynamics(BaseParameterDynamics):
    """
        A parameter dynamics that remains constant throughout the simulation.
    """
    kind: str = "constant"
    init_value: float

    def __init__(self, init_value: float):
        super().__init__()
        self.init_value = init_value

    def on_parameter_update(self, cell_type: CellType, step: int):
        return self.init_value