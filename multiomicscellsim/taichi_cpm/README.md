# Taichi Cell Potts Model (Taichi CPM)

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


## Getting Started

The main component is the `Simulation` class in `simulation.py`, which implements the CPM algorithm and manages the simulation state. 

### Cell Potts Model
The Cellular Potts Model is a lattice-based computational model used to simulate the collective behavior of cells. Each cell is represented as a collection of connected lattice sites (pixels) on a grid, and the model evolves over time based on energy minimization principles. 

At each time step, the model randomly selects two neighboring lattice sites (taken from the points of contact between cells or between a cell and the medium) and attempts to copy the state (cell ID) of one site to the other. The change is accepted or rejected based on the change that would occur in the system's total energy, which is computed based on various constraints and biases that model cell behaviors. For example, cells may have a preferred volume and perimeter, and the energy function will favor copies that go towards these preferred values.

### Simulation Class
The simulation is configured using a `SimConfig` dataclass that specifies parameters such as grid size, number of cell types, constraints, biases, etc. 
Think of it as the main entry point for running a simulation, and it used to define the "world" parameters in which the cells exist.

### Constraints and Biases

The `Constraint` class is the base class for all constraints and biases. Each specific constraint or bias (e.g., `VolumeConstraint`, `ChemotaxisBias`) corresponds to a new energy term in the CPM energy function, and it's governed by a `lambda_weight` parameter that determines its influence on the overall energy.

In this context, the difference between a constraint and a bias lies in how the term is defined.
- **Constraints**: These are more complex energy terms that typically define energy landscapes based on structural cell properties, such as volume and perimeter. 
- **Biases**: These are simpler energy terms that often represent directional influences on cell movement, such as chemotaxis or repulsion.

Constraints and biases, when enabled, are simulation-wide rules that determine how cells behave based on their properties. In simple terms, they are the "laws" that govern cell behavior in the whole simulation.

### Cell Types and Behaviours
Cells are represented by unique IDs on the grid, and each cell has a type that determines its desired properties and behaviors. The `CellType` dataclass is a container for defining these properties, such as target volume, target perimeter, and adhesion characteristics. To promote diversity in cell behavior, each `CellType` defines a list of behavior statistics that are used to initialize the individual preferred properties of each cell of that type.

To allow for different cells to have distinct behaviors, the simulation supports multiple cell types, and each `CellType` can be configured with a list of `Behaviour` instances. Each `Behaviour` represents the typical actions a cell of that type performs under certain conditions, such as growth and division. In practice, this means that each behavior has control over the *preferred properties* of the cell, which are then used by the constraints to compute the energy changes during the simulation. 
To implement this, the `Behaviour` class supports different `Dynamic` properties that can be used to implement time-dependent changes in cell properties, such as sinusal growth depending on the cell's age.

### Cell Initialization

To promote diversity in the initial cell configuration, the C

### Setting up and running a simulation

The simulation class and each Constraint and Behaviour has its corresponding pydantic dataclass in the `config` folder. This allows to build a configuration file in JSON or YAML format that can be loaded to create a simulation instance.

once a configuration file is created, a simulation can be run with the following code:

```python
from multiomicscellsim.taichi_cpm.simulation import Simulation

config = ...
sim = Simulation(config)
sim.run()

```

if real-time visualization is enabled in the configuration file, a simple GUI will be launched to preview the simulation state.


## Developer Guides

This module is implemented in Taichi, a high-performance programming language for numerical computation. Due to this, familiarity with Taichi is recommended for contributing to the codebase since it imposes certain restrictions on how code can be structured and executed. (e.g., no dynamic memory allocation, restricted use of Python features). These restrictions comes with the benefit of enabling high-performance execution on both CPU and GPU that allows for a faster evaluation of simulation configurations. If you are new to Taichi, consider reviewing the [Taichi documentation](https://docs.taichi-lang.org/) and tutorials to become familiar with its syntax and capabilities.

The codebase has a modular structure, with separate files for different components such as constraints, biases, and the main simulation loop. This modularity makes it easier to understand and implement new features that can be useful in different areas of research.

### How to add a new constraint

1. Create a new file in the `constraints` folder, e.g. `my_constraint.py`.
2. Derive a new class from `BaseConstraint` and implement the required methods.
3. Create a corresponding configuration dataclass in `config/constraints.py` and add to the `constraint_factory` method.
4. Add the new constraint to the `sim_config` incrementing the `energy_index`.


### Adding a new behaviour

1. Create a new Behaviour class that derives from `Behaviour` in `behaviours.py`.
2. Implement the `on_behaviour_update(self, cell_id: int)` method to define the behavior logic. This method is called at each simulation step before computing the energy changes. It receives the `cell_id` of the cell to which the behavior is applied and *it is responsible for updating the cell's preferred properties*. Please note that the proper way to update the cell's properties is by accessing the `self.sim.cell_id[cell_id]` object, which contains the cell's current properties. Also note that different behaviours can be applied to the same parameter, for that reason each behaviour has to list the parameters it modifies in the `influences` attribute. If needed, the behaviour can also access the full simulation state via `self.sim`, but it should only modify the properties of the cell it is applied to.
3. Create a corresponding configuration dataclass in `config/behaviours.py` and add to the `behaviour_factory` method.
4. Add the new behaviour to the desired `CellType` in the `sim_config`.

### Other notes

- Cells, CellTypes and Constraints references are kept in Taichi fields, which are statically typed structures similar to C arrays. This means that the number of cells and constraints must be defined at the start of the simulation and cannot be changed dynamically.
- Stroma (intracellular medium) is represented by cell ID 0, and it is treated as a special cell type with id 0. At the current state it has no special properties nor behaviours, but this should be kept in mind when interacting with the simulation state (i.e., The first visible cell is in position 1).
 