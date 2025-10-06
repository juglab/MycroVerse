
from pydantic import BaseModel, Field
from multiomicscellsim.taichi_cpm.constraints.perimeter import PerimeterConstraint
from multiomicscellsim.taichi_cpm.constraints.volume import VolumeConstraint
from multiomicscellsim.taichi_cpm.constraints.adhesion import AdhesionConstraint
from multiomicscellsim.taichi_cpm.constraints.orientation import PolarizationConstraint, AnisotropyOrientationConstraint
from multiomicscellsim.taichi_cpm.constraints.biases import InvasionPenaltyBias, ChemotaxisBias, RepulsionBias, PolarizationBias
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
    
class AdhesionConstraintConfig(ConstraintConfig):
    """
        Configuration for adhesion constraint.
    """
    name: str = "AdhesionConstraint"

class InvasionPenaltyConstraintConfig(ConstraintConfig):
    """
        Configuration for invasion penalty constraint.
    """
    name: str = "InvasionPenaltyConstraint"

class ChemotaxisConstraintConfig(ConstraintConfig):
    """
        Configuration for chemotaxis constraint.
    """
    name: str = "ChemotaxisConstraint"

class RepulsionConstraintConfig(ConstraintConfig):
    """
        Configuration for repulsion constraint.
    """
    name: str = "RepulsionConstraint"

class PolarizationBiasConfig(ConstraintConfig):
    """
        Configuration for polarization bias constraint.
    """
    name: str = "PolarizationBias"

class PolarizationConstraintConfig(ConstraintConfig):
    """
        Configuration for polarization constraint constraint.
    """
    name: str = "PolarizationConstraint"

class AnisotropyOrientationConstraintConfig(ConstraintConfig):
    """
        Configuration for orientation anisotropy constraint.
    """
    name: str = "AnisotropyOrientationConstraint"
    lambda_anisotropy: float = Field(..., description="Weight of the anisotropy term.")
    lambda_orientation: float = Field(..., description="Weight of the orientation term.")


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
    elif isinstance(config, AdhesionConstraintConfig):
        return AdhesionConstraint(sim, config.energy_index, config.lambda_weight)
    elif isinstance(config, InvasionPenaltyConstraintConfig):
        return InvasionPenaltyBias(sim, config.energy_index, config.lambda_weight)
    elif isinstance(config, ChemotaxisConstraintConfig):
        return ChemotaxisBias(sim, config.energy_index, config.lambda_weight)
    elif isinstance(config, RepulsionConstraintConfig):
        return RepulsionBias(sim, config.energy_index, config.lambda_weight)
    elif isinstance(config, PolarizationConstraintConfig):
        return PolarizationConstraint(sim, config.energy_index, config.lambda_weight)
    elif isinstance(config, PolarizationBiasConfig):
        return PolarizationBias(sim, config.energy_index, config.lambda_weight)
    elif isinstance(config, AnisotropyOrientationConstraintConfig):
        return AnisotropyOrientationConstraint(sim, config.energy_index, config.lambda_weight, config.lambda_anisotropy, config.lambda_orientation)
    else:
        raise ValueError(f"Unknown constraint type: {config.name}")
