
from pydantic import BaseModel, Field
from multiomicscellsim.taichi_cpm.constraints.perimeter import PerimeterConstraint
from multiomicscellsim.taichi_cpm.constraints.volume import VolumeConstraint
from multiomicscellsim.taichi_cpm.constraints.base import Constraint



class ConstraintConfig(BaseModel):
    """
        Configuration for constraints in the simulation.
    """
    name: str = Field(..., description="Name of the constraint.")
    lambda_weight: float = Field(..., description="Weight of the constraint.")
    energy_index: int = Field(..., description="Index of the energy term associated with the constraint.")


class PerimeterConstraintConfig(ConstraintConfig):
    """
        Configuration for perimeter constraint.
    """
    name: str = "PerimeterConstraint"

class VolumeConstraintConfig(ConstraintConfig):
    """
        Configuration for volume constraint.
    """
    name: str = "VolumeConstraint"
    
def constraint_factory(config: ConstraintConfig, sim, *args, **kwargs) -> Constraint:
    """
        Factory function to create constraint instances based on the configuration.
        Args:
            config: The configuration object for the constraint.
            sim: The simulation instance to which the constraint will be applied.
    """
    if isinstance(config, PerimeterConstraintConfig):
        return PerimeterConstraint(sim, config.energy_index, config.lambda_weight)
    elif isinstance(config, VolumeConstraintConfig):
        return VolumeConstraint(sim, config.energy_index, config.lambda_weight)
    else:
        raise ValueError(f"Unknown constraint type: {config.name}")
