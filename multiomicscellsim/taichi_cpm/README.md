# Taichi Cellular Potts Model (Taichi CPM)

This package contains a Taichi-based implementation of the Cellular Potts Model (CPM) for simulating cell behaviors and interactions in a 2D grid. The simulation includes various constraints and biases to model virtual cell dynamics.

## Features

- **Cell Growth and Division**: Cells can grow and divide based on defined parameters.
- **Cell Types**: Supports multiple cell types with distinct properties.
- **Constraints**: Includes volume, perimeter, and adhesion to regulate cell shapes
- **Biases**: Supports chemotaxis, invasion penalty, repulsion, and polarization biases to influence cell movement
- **Taichi Backend**: Utilizes the Taichi programming language for high-performance real-time simulations on both CPU and GPU.

## WIP Features
- **Genetic Expression**: Basic framework for simulating gene expression dynamics within cells.
- **Orientation Constraints**: Support for polarization and anisotropy orientation constraints to model cell alignment and shape anisotropy.
- **Cell Differentiation**: Mechanisms for changing cell types after division.
- **Chemokine Dynamics**: Simulation of chemokine fields affecting cell movement.
- **Cell Signaling**: Basic framework for cell-cell signaling interactions via diffusible factors.
- **Subcellular Representation**: Integration with reaction-diffusion models for subcellular content representation.
- **Cell Cycle Modeling**: Framework for simulating cell cycle phases and transitions.


## Developer Guides

### How to add a new constraint

1. Create a new file in the `constraints` folder, e.g. `my_constraint.py`.
2. Derive a new class from `BaseConstraint` and implement the required methods.
3. Create a corresponding configuration dataclass in `config/constraints.py` and add to the `constraint_factory` method.
4. Add the new constraint to the `sim_config` incrementing the `energy_index`.