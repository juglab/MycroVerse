
from multiomicscellsim.taichi_cpm.simulation import Simulation
from multiomicscellsim.taichi_cpm.config.constraints import PerimeterConstraintConfig, \
                                                            VolumeConstraintConfig, \
                                                            AdhesionConstraintConfig, \
                                                            InvasionPenaltyConstraintConfig, \
                                                            ChemotaxisConstraintConfig, \
                                                            RepulsionConstraintConfig, \
                                                            PolarizationBiasConfig, \
                                                            PolarizationConstraintConfig, \
                                                            AnisotropyOrientationConstraintConfig

from multiomicscellsim.taichi_cpm.config.cell_types import CellTypeConfig
from multiomicscellsim.taichi_cpm.config.behaviours import  EllipticPerimeterBehaviourConfig, \
                                                            MitosisAgeVolumeBehaviourConfig, \
                                                            AdhesionBehaviourConfig , \
                                                            VolumeBehaviourConfig
from multiomicscellsim.taichi_cpm.config.dynamics import SinusoidalDynamicsConfig


sin_dynamics = SinusoidalDynamicsConfig(
                                                                                      offset=1400,
                                                                                      amplitude=900,
                                                                                      frequency=0.1,
                                                                                      phase=0.0)

sim_config = {
    "random_seed": 42,
    "size": 512,
    "max_cell_types": 10,
    "max_cells": 500,
    "temperature": 1,
    "selection_probability": 1,
    "chemokine_seed": 0,
    "chemokine_noise_scale": 5,
    "constraints": [
        VolumeConstraintConfig(lambda_weight=1e-3, energy_index=0),
        PerimeterConstraintConfig(lambda_weight=1e-3, energy_index=1),
        AdhesionConstraintConfig(lambda_weight=1, energy_index=2),
        InvasionPenaltyConstraintConfig(lambda_weight=1, energy_index=3),
        ChemotaxisConstraintConfig(lambda_weight=50, energy_index=4),
        RepulsionConstraintConfig(lambda_weight=0, energy_index=5),
        #PolarizationBiasConfig(lambda_weight=1, energy_index=6),
        # AnisotropyOrientationConstraintConfig(lambda_weight=5, 
        #                                       lambda_anisotropy=1, 
        #                                       lambda_orientation=1, 
        #                                       energy_index=6)
    ],
    "cell_types": [
        CellTypeConfig( 
            name="Type 1",
            preferred_volume_stats=[0.005, 0.0001],
            preferred_anisotropy_stats=[0.8, 0.001],
            preferred_orientation_stats=[[1.0, 0.1], [0.0, 0.1]],
            mitosis_age_stats=[10*10, 50],
            behaviours=[
                            AdhesionBehaviourConfig(j_adhesion_stroma=0.0, 
                                                    j_adhesion_other=100.0),
                            VolumeBehaviourConfig(dynamics=sin_dynamics),
                            EllipticPerimeterBehaviourConfig(dynamics=None),
                            MitosisAgeVolumeBehaviourConfig(dynamics=None, 
                                                            sigmoid_slope=0.01, 
                                                            probability_scale=0.
                                                            ),
                            
                       ]
        ),
        CellTypeConfig(
            name="Type 2",
            preferred_volume_stats=[0.003, 0.0001],
            preferred_anisotropy_stats=[0.8, 0.001],
            preferred_orientation_stats=[[1.0, 0.1], [0.0, 0.1]],
            mitosis_age_stats=[60*10, 50],
            behaviours=[    
                            AdhesionBehaviourConfig(j_adhesion_stroma=0.0, 
                                                    j_adhesion_other=4.0),
                            VolumeBehaviourConfig(),
                            EllipticPerimeterBehaviourConfig(dynamics=None),
                            MitosisAgeVolumeBehaviourConfig(dynamics=None, 
                                                            sigmoid_slope=0.01, 
                                                            probability_scale=0.1,
                                                            )
                       ]
        ),
        CellTypeConfig(
            name="Type 3",
            preferred_volume_stats=[0.01, 0.0001],
            preferred_anisotropy_stats=[0.2, 0.001],
            preferred_orientation_stats=[[1.0, 0.1], [0.0, 0.1]],
            mitosis_age_stats=[60*10, 50],
            behaviours=[    
                            AdhesionBehaviourConfig(j_adhesion_stroma=0.0, 
                                                    j_adhesion_other=4.0),
                            VolumeBehaviourConfig(),
                            EllipticPerimeterBehaviourConfig(dynamics=None),
                            MitosisAgeVolumeBehaviourConfig(dynamics=None, 
                                                            sigmoid_slope=0.01, 
                                                            probability_scale=0.1,
                                                            )
                       ]
        ),
    ]
}

sim = Simulation(sim_config)
sim.run()