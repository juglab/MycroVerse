from pydantic import BaseModel, Field
from typing import List
from multiomicscellsim.taichi_cpm.config.dynamics import BaseParameterDynamicsConfig, dynamics_factory
from multiomicscellsim.taichi_cpm.behaviours import BaseBehaviour, EllipticPerimeter, MitosisAgeVolumeCopyBehaviour


class BaseBehaviourConfig(BaseModel):
    name: str = Field(description="Description name of the behaviour")
    influences: List[str]
    dynamics: BaseParameterDynamicsConfig | None = Field(description="Configuration for the dynamics of the behaviour")

class EllipticPerimeterConfig(BaseBehaviourConfig):
    name: str = "approximate_ellipse"
    influences: List[str] = ["preferred_perimeter"]

class MitosisAgeVolumeBehaviourConfig(BaseBehaviourConfig):
    """
        The cell tries to divide when it reaches a certain age and volume.
    """
    name: str = ""
    influences: List[str] = ["mitosis_prob_age", "mitosis_prob_volume", "mitosis_probability"]
    sigmoid_slope: float = Field(0.1, description="Slope of the sigmoid function to compute mitosis probability")
    probability_scale: float = Field(1.0, description="Scaling factor for the mitosis probability")

def behaviour_factory(config: BaseBehaviourConfig, sim, *args, **kwargs) -> BaseBehaviour:
    """
        Factory function to create a Behaviour from a BehaviourConfig.

        Args:
            config (BaseBehaviourConfig): The configuration for the Behaviour.
            sim: The simulation instance.
    """
    dynamics = dynamics_factory(config.dynamics, sim, *args, **kwargs)
   
    if isinstance(config, EllipticPerimeterConfig):
        return EllipticPerimeter(sim, dynamics)
    elif isinstance(config, MitosisAgeVolumeBehaviourConfig):
        return MitosisAgeVolumeCopyBehaviour(sim, dynamics, config.sigmoid_slope, config.probability_scale)
    else:
        print(f"Unknown behaviour config type: {type(config)} {config}")
        raise ValueError(f"Unknown behaviour config type: {type(config)}")
    