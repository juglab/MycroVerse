from pydantic import BaseModel, Field
from multiomicscellsim.taichi_cpm.config.behaviours import BaseBehaviourConfig
from multiomicscellsim.taichi_cpm.entities import CellType
from typing import List
import taichi as ti

class CellTypeConfig(BaseModel):
    """
        A celltype defines the base statistics
        for constraint parameters, defining the cell behaviours
    """
    name: str = Field(description="Name of the cell type")
    #behaviours: List[BaseBehaviourConfig] = Field(description="List of BehavioursConfig to govern preferred cell parameters")
    j_adhesion_stroma: float = Field(description="Penalty coefficient for adhesion energy towards the stroma")
    j_adhesion_other: float = Field(description="Penalty coefficient for adhesion towards other cells")
    preferred_volume_stats: List[float] = Field(description="Mean and std for preferred volume")
    preferred_anisotropy_stats: List[float] = Field(description="Mean and std for preferred anisotropy")
    preferred_orientation_stats: List[List[float]] = Field(description="Mean and std for preferred orientation")
    mitosis_age_stats: List[float] = Field(description="Mean and std for mitosis age")
    behaviours: List[BaseBehaviourConfig] = Field(description="List of BehavioursConfig to govern preferred cell parameters")

def celltype_factory(id: int, ct: CellType, config: CellTypeConfig) -> None:
    """
        Factory function to create a CellType from a CellTypeConfig.

        Args:
            ct (CellType): The CellType instance to be configured.
            config (CellTypeConfig): The configuration for the CellType.
    """
    ct.id = id
    ct.j_adhesion_stroma = config.j_adhesion_stroma
    ct.j_adhesion_other = config.j_adhesion_other
    ct.preferred_volume_stats = ti.Vector(arr=config.preferred_volume_stats)
    ct.preferred_anisotropy_stats = ti.Vector(arr=config.preferred_anisotropy_stats)
    ct.preferred_orientation_stats = ti.Vector(arr=config.preferred_orientation_stats)
    ct.mitosis_age_stats = ti.Vector(arr=config.mitosis_age_stats)
    # TODO: Implement behaviours initialization
    # Since we cannot embed a list of behaviours in CellType (Taichi limitation),
    # we will keep them in the simulation class, mapping cell types to their behaviours.