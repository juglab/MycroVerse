from pydantic import BaseModel, Field
from typing import List
from multiomicscellsim.taichi_cpm.dynamics import ConstantParameterDynamics

class BaseParameterDynamicsConfig(BaseModel):
    name: str = Field(description="Description name of the dynamics")

class ConstantDynamicsConfig(BaseParameterDynamicsConfig):
    name: str = "constant"

def dynamics_factory(config: BaseParameterDynamicsConfig, sim, *args, **kwargs):
    """
        Factory function to create a Dynamics from a DynamicsConfig.

        Args:
            config (BaseParameterDynamicsConfig): The configuration for the Dynamics.
            sim: The simulation instance.
    """
    if isinstance(config, ConstantDynamicsConfig):
        return ConstantParameterDynamics(init_value=1.0)
        

    
    
