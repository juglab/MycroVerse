
from abc import ABC, abstractmethod
from typing import List
from multiomicscellsim.taichi_cpm.entities import CellType
import taichi as ti

class BaseParameterDynamics(ABC):
    """
        Base class for parameter dynamics.
        A parameter dynamics implement a strategy to update preferred parameters of a cell over time.
    """
    kind: str
    def __init__(self):
        pass

    @abstractmethod
    def on_parameter_update(self, cell_id: int, sim, current_value: int, step :int):
        pass

class SinusoidalParameterDynamics(BaseParameterDynamics):
    """
        A parameter dynamics that varies sinusoidally over time.
    """
    kind: str = "sinusoidal"
    amplitude: float
    frequency: float
    phase: float
    offset: float

    def __init__(self, amplitude: float, frequency: float, phase: float, offset: float):
        super().__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    @ti.func
    def on_parameter_update(self, cell_id: int, sim, current_value: int, step :int):
        return self.offset + self.amplitude * ti.sin(self.frequency * step + self.phase)

class ConstantParameterDynamics(BaseParameterDynamics):
    """
        A parameter dynamics that remains constant throughout the simulation.
    """
    kind: str = "constant"

    def __init__(self):
        super().__init__()

    def on_parameter_update(self, cell_id: int, sim, current_value: int, step :int):
        return current_value