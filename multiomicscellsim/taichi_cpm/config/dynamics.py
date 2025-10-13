from pydantic import BaseModel, Field
from typing import List
from multiomicscellsim.taichi_cpm.dynamics import ConstantParameterDynamics, SinusoidalParameterDynamics

class BaseParameterDynamicsConfig(BaseModel):
    name: str = Field(description="Description name of the dynamics")

class ConstantDynamicsConfig(BaseParameterDynamicsConfig):
    name: str = "constant"

class SinusoidalDynamicsConfig(BaseParameterDynamicsConfig):
    name: str = "sinusoidal"
    amplitude: float = Field(..., description="Amplitude of the sinusoidal variation")
    frequency: float = Field(..., description="Frequency of the sinusoidal variation")
    phase: float = Field(..., description="Phase shift of the sinusoidal variation")
    offset: float = Field(..., description="Offset of the sinusoidal variation")

def dynamics_factory(config: BaseParameterDynamicsConfig, sim, *args, **kwargs):
    """
        Factory function to create a Dynamics from a DynamicsConfig.

        Args:
            config (BaseParameterDynamicsConfig): The configuration for the Dynamics.
            sim: The simulation instance.
    """
    if isinstance(config, ConstantDynamicsConfig):
        return ConstantParameterDynamics()
    elif isinstance(config, SinusoidalDynamicsConfig):
        return SinusoidalParameterDynamics(
            amplitude=config.amplitude,
            frequency=config.frequency,
            phase=config.phase,
            offset=config.offset
        )
    else:
        print(f"Unknown dynamics config type: {type(config)} {config}")
        raise ValueError(f"Unknown dynamics config type: {type(config)}")
        

    
    
