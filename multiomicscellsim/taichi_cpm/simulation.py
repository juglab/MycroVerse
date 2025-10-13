import taichi as ti
import math
import numpy as np
from perlin_numpy import generate_perlin_noise_2d
from typing import List
from entities import Cell, CellPoint, CellType
from utils import point_in_polygon, get_polygon_edges, local_perimeter, compute_eigenvectors_and_eigenvalues
from multiomicscellsim.taichi_cpm.config.constraints import constraint_factory
from multiomicscellsim.taichi_cpm.config.cell_types import celltype_factory
from multiomicscellsim.taichi_cpm.config.static import MAX_ENERGY_TERMS
from multiomicscellsim.taichi_cpm.config.behaviours import behaviour_factory

@ti.data_oriented
class Simulation():
    def __init__(self, sim_config):
        
        # Set random seed for reproducibility
        self.config = sim_config
        self.seed = sim_config.get("random_seed", 0)
        self.random = np.random.default_rng(self.seed)
        ti.init(arch=ti.cpu, debug=False, random_seed=self.seed, cpu_max_num_threads=1) # cpu_max_num_threads=1 for reproducibility

        # Simulation parameters
        self.size = sim_config["size"]
        self.max_cells=sim_config["max_cells"]
        self.temperature = sim_config["temperature"]
        self.select_prob = ti.field(dtype=float, shape=())
        self.select_prob[None] = sim_config.get("selection_probability", 0.5) # Probability of selecting a cell for copying

        # Define constraints
        assert len(sim_config["constraints"]) <= MAX_ENERGY_TERMS, f"Number of constraints exceeds MAX_ENERGY_TERMS ({MAX_ENERGY_TERMS})"
        
        self.constraints = []        
        for constraint_config in sim_config["constraints"]:
            new_constraint = constraint_factory(constraint_config, self)
            self.constraints.append(new_constraint)

        self.n_constraints = len(self.constraints)

        # Chemicals parameters
        self.chemokine_seed = sim_config.get("chemokine_seed", 0)
        self.chemokine_scale = sim_config.get("chemokine_noise_scale", 6.0)

        # Simulation state
        self.sim_pause = ti.field(dtype=int, shape=())
        self.display_mode = ti.field(dtype=int, shape=())
        self.cell_selected = ti.field(dtype=int, shape=())

        self.gui = ti.GUI(name="MycroVerse", res=self.size)
        self.render_grid = ti.field(dtype=float, shape=(self.size, self.size, 3))
        
        # Field mapping behaviours (index in self.behaviours) to cell types id.
        # Initialized in setup_celltypes_and_behaviours
        self.behaviour_to_celltype = None
        self.setup_celltypes_and_behaviours()
        

        print(self.behaviours)
        self.reinit()

    def setup_celltypes_and_behaviours(self):
        """
            Populate the cell_types field with the configured cell types.
        """

        n_celltypes = len(self.config["cell_types"])

        self.behaviours = []
        behaviours_configs = []
    
        # We reserve slot 0 for the background to support stroma properties
        self.cell_types = CellType.field(shape=(n_celltypes+1,))
        for c, ct_cfg in enumerate(self.config["cell_types"]):

            celltype_factory(c+1, self.cell_types[c+1], ct_cfg)
            # Maintains a list of behaviours for each cell type 
            # (we cannot embed them in CellType because Taichi does not support dynamic lists and classes in dataclasses)
            ct_behaviours = ct_cfg.behaviours
            behaviours_configs.append(ct_behaviours)

        # Convert to Taichi fields
        self.n_behaviours = sum(len(bh_cfgs) for bh_cfgs in behaviours_configs)
        self.behaviour_to_celltype = ti.field(dtype=int, shape=(self.n_behaviours,))
        for ct_idx, bh_cfgs in enumerate(behaviours_configs):
            for bh_cfg in bh_cfgs:
                # Create behaviour instance
                behaviour = behaviour_factory(bh_cfg, self)
                self.behaviours.append(behaviour)
                self.behaviour_to_celltype[len(self.behaviours)-1] = ct_idx + 1 # +1 because 0 is the background

    def reinit(self):
        self.grid = CellPoint.field(shape=(self.size, self.size))
        self.chemokine_grid = ti.field(dtype=float, shape=(self.size, self.size))
        self.cells = Cell.field(shape=(self.max_cells+1,))
        self.n_cells = ti.field(dtype=int, shape=())
        self.n_cells[None] = 1 # First cell is the background
        self.chemokine_setup()

        # TODO: Stochastic cell generation
        n_celltypes = len(self.config["cell_types"])
        # Place cells on a grid 
        for ct_idx in range(1, n_celltypes + 1):
            print(f"Creating initial cell of type {ct_idx}")
            x_c = self.random.uniform(0.2, 0.8)
            y_c = self.random.uniform(0.2, 0.8)
            self.create_cell(x_c, y_c, ct_idx)
        # Recalc all params before starting simulation
        self.update_grid_params()
    
    def chemokine_setup(self):
        """
            Pre-compute all numpy-related components that depends on a different seed and does not support external RNG state.
        """
        # Save current global RNG state
        state = np.random.get_state()
        perlin_scale = self.size // int((2**self.chemokine_scale))

        np.random.seed(self.chemokine_seed)
        noise = generate_perlin_noise_2d((self.size, self.size), (perlin_scale, perlin_scale))
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        self.chemokine_grid.from_numpy(noise)

        # Restore original global RNG state
        np.random.set_state(state)

    @ti.kernel
    def draw_polygon_on_grid(self, polygon:ti.template(), cell_id:int, cell_type:int): # type: ignore
        for i, j in self.grid:
            if point_in_polygon(i, j, polygon=polygon):
                self.grid[i,j].cell_id = cell_id
                self.grid[i,j].cell_type = cell_type

    def create_cell(self, xc: float, yc: float, cell_type:int):
        """
            Create a new cell at the given normalized coordinates (0-1) with the specified cell type.
            The cell ID is automatically assigned.
            The cell is drawn as a circle with a fixed radius.
            Args:
                xc (float): Normalized x-coordinate (0-1) for the cell center.
                yc (float): Normalized y-coordinate (0-1) for the cell center.
                cell_type (int): The type of the cell to be created.
        """

        # Sample params from stats
        cell_id = self.n_cells[None]
        self.n_cells[None] += 1
        self.cells[cell_id].cell_id = cell_id
        # Set Cell parameters according to the cell type
        self.cell_types[cell_type].sample_cell(self.cells[cell_id], self)
        # Draw a circular cell
        radius = 0.02
        n_edges = 16
        verts = ti.Vector.field(n=2, dtype=int, shape=(n_edges,))
        get_polygon_edges(xc, yc, n_edges=n_edges, radius=radius, grid_size=self.size, verts=verts)
        self.draw_polygon_on_grid(polygon=verts, cell_id=cell_id, cell_type=cell_type)
        

    @ti.func
    def is_out_of_bounds(self, i:int, j:int) -> int:
        return ti.cast(i< 0 or i >= self.size or j<0 or j>=self.size, int)

    @ti.func
    def pick_random_neighbor(self, i, j) -> ti.Vector:
        neigh = ti.Vector([i, j])

        while neigh[0] == i and neigh[1] == j:
            # Generates an offset between [-1, 0, 1] for i and j
            oi = ti.floor(3*ti.random(), int) - 1
            oj = ti.floor(3*ti.random(), int) - 1
            ni = i + oi
            nj = j + oj
            if not self.is_out_of_bounds(ni, nj) and self.grid[ni, nj].cell_id != self.grid[i, j].cell_id:
                neigh[0] = ni
                neigh[1] = nj
        return neigh

    @ti.kernel
    def update_grid_params(self):
        """
            Update all grid and cell parameters based on the current grid state.
            This includes recalculating the local perimeter, current volume and current perimeter of each cell.
        """

        # Reset all cell parameters
        for c in self.cells:
            if self.cells[c].cell_type > 0:
                self.cells[c].zero_current_parameters()

        # Accumulate grid parameters (operations on pixels)
        for i, j in self.grid:
            self.grid[i, j].local_perimeter = local_perimeter(self, i, j, self.grid[i, j].cell_id)
            self.grid[i, j].is_membrane = int(self.grid[i, j].local_perimeter > 0)  
            self.grid[i, j].selected_as_src = 0
            self.grid[i, j].selected_as_trg = 0
            self.grid[i, j].copy_into.fill(0)
            self.grid[i, j].copy_from.fill(0)
            if self.grid[i, j].cell_id > 0:
                ti.atomic_add(self.cells[self.grid[i, j].cell_id].current_volume, 1.0)
                ti.atomic_add(self.cells[self.grid[i, j].cell_id].current_perimeter, self.grid[i, j].local_perimeter)
                # Use center as accumulator temporarily
                ti.atomic_add(self.cells[self.grid[i, j].cell_id].center.x, float(i))
                ti.atomic_add(self.cells[self.grid[i, j].cell_id].center.y, float(j))
        
        # Calculate center of mass
        for c in self.cells:
            if self.cells[c].current_volume > 0:
                # Center of mass (Assume center contains the sum of positions)
                self.cells[c].center /= self.cells[c].current_volume
                
        # Second pass (computations that depends on center of mass / preferred perimeter / etc)
        for i, j in self.grid:
            if self.grid[i, j].cell_id > 0:
                # Update covariance matrix
                diff = ti.Vector([float(i), float(j)]) - self.cells[self.grid[i, j].cell_id].center

                # Using outer product to accumulate covariance
                ti.atomic_add(self.cells[self.grid[i, j].cell_id].covariance_matrix, ti.Matrix([[diff.x * diff.x, diff.x * diff.y],
                                                                                            [diff.x * diff.y, diff.y * diff.y]]))
                
        for c in self.cells:
            if self.cells[c].current_volume > 0:
                # Normalize covariance matrix
                self.cells[c].covariance_matrix /= self.cells[c].current_volume
                # Compute current orientation as the longest axis of the covariance matrix
                self.cells[c].maj_axis, self.cells[c].min_axis, self.cells[c].max_eigenvalue, self.cells[c].min_eigenvalue = compute_eigenvectors_and_eigenvalues(self.cells[c].covariance_matrix)
                self.cells[c].current_anisotropy = (self.cells[c].max_eigenvalue - self.cells[c].min_eigenvalue) / (self.cells[c].max_eigenvalue + self.cells[c].min_eigenvalue + 1e-6)

            # Update behavior
            for behav_id in ti.static(range(self.n_behaviours)):
                if self.behaviour_to_celltype[behav_id] == self.cells[c].cell_type:
                    self.behaviours[behav_id].on_behaviour_update(cell_id=c) # FIXME: pass the step number?
                    #print(f"Updated Cell {self.cells[c].cell_id} - Preferred Perimeter: {self.cells[c].preferred_perimeter}, Current Perimeter: {self.cells[c].current_perimeter}")

            if c > 0 and self.cells[c].cell_type > 0:

                # Recompute energy terms
                for cid in ti.static(range(self.n_constraints)):
                    self.cells[c].current_energy_terms[self.constraints[cid].energy_index] = self.constraints[cid].calculate_current_cell_energy(c)


    @ti.kernel
    def do_copy(self):
        """
            Perform the copy of the selected pixels based on the calculated energy deltas and the Metropolis criterion.
            If the copy is accepted, update the target pixel's cell_id and cell_type.
        """

        for i, j in self.grid:
            if self.grid[i, j].selected_as_src:
                t_i, t_j = ti.cast(self.grid[i, j].copy_into[0], int), ti.cast(self.grid[i, j].copy_into[1], int)
                delta_E = self.grid[i, j].copy_energy_delta
                accept = False

                if delta_E <= 0:
                    accept = True
                else:
                    prob = ti.exp(-delta_E / self.temperature)
                    if ti.random() < prob:
                        accept = True

                if accept: 
                    self.grid[t_i, t_j].cell_type = self.grid[i, j].cell_type
                    self.grid[t_i, t_j].cell_id = self.grid[i, j].cell_id

                # Always reset regardless of acceptance
                self.grid[t_i, j].selected_as_src = 0
                self.grid[t_i, j].selected_as_trg = 0
                self.grid[t_i, j].copy_energy_delta = 0

    @ti.kernel
    def trigger_mitosis(self):
        """
            Check for cells that should undergo mitosis based on their mitosis probability and anisotropy threshold.
            If a cell is set to divide, mark it for splitting in the next step.
        """

        for cell_id in range(self.n_cells[None]):
            if cell_id > 0 and self.cells[cell_id].should_split > 0:
                new_cell_id = self.n_cells[None]
                
                if new_cell_id < self.max_cells:    
                    # Split from center to a random direction
                    old_center = self.cells[cell_id].center
                    split_direction = self.cells[cell_id].maj_axis.normalized() # Use the major axis as the split direction
                    # Taichi does not support dynamic nested fors......
                    new_current_size = 0
                    for i in range(self.size):
                        for j in range(self.size):    
                            if self.grid[i, j].cell_id == cell_id:
                                pos = ti.Vector([float(i), float(j)])
                                # Assign new cell id to one side of the splitted cell
                                if (pos - old_center).dot(split_direction) > 0:
                                    self.grid[i, j].cell_id = new_cell_id
                                    ti.atomic_add(new_current_size, 1)
                    
                    self.cells[cell_id].should_split = 0
                    ti.atomic_add(self.n_cells[None], 1)
                    
                    # Initialize new cell parameters by calling the cell behaviours
                    self.cells[new_cell_id].cell_id = new_cell_id
                    for behav_id in ti.static(range(self.n_behaviours)):
                        if self.behaviour_to_celltype[behav_id] == self.cells[cell_id].cell_type:
                            self.behaviours[behav_id].on_mitosis(mother_id=cell_id, daughter_id=new_cell_id)

   
    @ti.kernel
    def select_potential_copies(self):
        """
            Randomly select potential source and target pixels for copying.
            A pixel can be selected as a source if it is a membrane pixel and not already selected
            as a source or target. The target pixel is chosen randomly from the neighbors of the source pixel.
        """
        for i, j in self.grid:
            if ti.random() < self.select_prob[None] and \
               self.grid[i, j].is_membrane and \
               not self.grid[i, j].selected_as_trg and \
               not self.grid[i, j].selected_as_src:
                
                # Pick a neighbor
                neigh = self.pick_random_neighbor(i, j)

                # Random swap direction
                s_i, s_j = i, j
                t_i, t_j = neigh[0], neigh[1]
                if ti.random() < .5:
                    s_i, s_j = neigh[0], neigh[1]
                    t_i, t_j = i, j                  
                
                # Avoid race conditions
                if not self.grid[s_i, s_j].selected_as_src and \
                   not self.grid[s_i, s_j].selected_as_trg and \
                   not self.grid[t_i, t_j].selected_as_src and \
                   not self.grid[t_i, t_j].selected_as_trg:

                    # Mark as selected
                    self.grid[s_i, s_j].selected_as_src = 1
                    self.grid[s_i, s_j].copy_into = ti.Vector([t_i, t_j], int)
                    self.grid[t_i, t_j].selected_as_trg = 1
                    self.grid[t_i, t_j].copy_from = ti.Vector([s_i, s_j], int)
    
    @ti.kernel
    def calc_energy(self):
        """
            Calculate the energy of the potential copies.
            For each selected source pixel, calculate the energy gain of copying it into its target pixel.
            
            We consider two kinds of energy terms:
            - Energy-deltas that can be computed locally (i.e. based on the local neighborhood of the source and target pixels)
            - Bias terms that depends on the cell's properties (e.g. chemotaxis, polarization, etc)
        """

        for i, j in self.grid:
            if self.grid[i, j].selected_as_src:
               
                t_i, t_j = ti.cast(self.grid[i, j].copy_into[0], int), ti.cast(self.grid[i, j].copy_into[1], int)
 
                # Energy deltas are calculated in the corresponding constraint classes and accumulated in the .copy_energy_delta field
                # Warning: This means that we cannot add new constraints at runtime, because Taichi needs to compile them statically
                for cid in ti.static(range(self.n_constraints)):
                    ti.atomic_add(self.grid[i, j].copy_energy_delta, self.constraints[cid].calculate_energy_delta(s_i=i, s_j=j, t_i=t_i, t_j=t_j))
                
                # If the cell is selected and either source or target, print debug info
                #print(f"Selected Cell: {self.grid[i, j].cell_id} ({i}, {j}) -> ({t_i}, {t_j}). {self.cell_selected}")
                if self.cell_selected[None] > 0 and (self.cell_selected[None] == self.grid[i, j].cell_id or self.cell_selected[None] == self.grid[t_i, t_j].cell_id):
                   print(chr(27) + "[2J")
                   print(f"Cell {self.cell_selected[None]} Selected. Copying from {i}, {j} to {t_i}, {t_j}")
                   print(f"Copy from {i}, {j} ({self.grid[i, j].cell_id}) to {t_i}, {t_j} ({self.grid[t_i, t_j].cell_id})")
                   
    def cpm_step(self):
        # Sources and targets are stored into .selected* and .copy_*
        self.select_potential_copies()
        # Calculate energy deltas for the selected pixels (stored in .copy_energy_delta)
        self.calc_energy()
        # Perform the copy based on the Metropolis criterion
        self.do_copy()
        # Update cell parameters from grid state and behaviours
        self.update_grid_params()
    
    @ti.kernel
    def check_mitosis(self):
        """
            Check if any cell should split based on its current volume.
            If the current volume is greater than the preferred volume, mark it for splitting.
        """
        # TODO: Move this into a behaviour
        for c in self.cells:
            if self.cells[c].cell_type >= 1:
                if ti.random() < self.cells[c].mitosis_probability:
                    self.cells[c].should_split = 1

    @ti.kernel
    def age_cells(self):
        for c in self.cells:
            if self.cells[c].cell_type >= 1 and c > 0:
                self.cells[c].current_age += 1.0

    def biology_events(self):
        """
            Handle biological events like mitosis.
        """
        self.age_cells()
        self.check_mitosis()
        self.trigger_mitosis()
        self.update_grid_params()


    def update(self):
        self.cpm_step()
        self.biology_events()
    
    @ti.kernel
    def render(self):
        """
            Populate the render_grid with the current state of the simulation.
        """
        mode = self.display_mode[None]
        # Maximum computed values to display across cells
        for i, j in self.grid:
            # Plot membrane in green
            if self.grid[i, j].is_membrane:
                self.render_grid[i, j, 0] = 0.0
                self.render_grid[i, j, 1] = 1.0
                self.render_grid[i, j, 2] = 0.0
            else:
                if self.grid[i, j].cell_id > 0:
                    if mode == 0:
                        self.render_grid[i, j, 0] = self.grid[i, j].cell_id / self.n_cells[None]
                        self.render_grid[i, j, 1] = self.grid[i, j].local_perimeter / 8
                        self.render_grid[i, j, 2] = 0.0
                    if mode == 1:
                        self.render_grid[i, j, 0] = self.cells[self.grid[i, j].cell_id].mitosis_probability * 10.0
                        self.render_grid[i, j, 2] = 0.0
                else:
                    # Display chemokine
                    self.render_grid[i, j, 2] = self.chemokine_grid[i, j]

    def draw(self):
        self.gui.clear()
        self.render_grid.fill(0)
        self.render()
        self.gui.set_image(self.render_grid)
        # Debug: Draw orientation vectors
        if self.display_mode[None] == 2:
            self.gui.arrows(orig=self.cells.center.to_numpy()/self.size, direction=self.cells.maj_axis.to_numpy()*self.cells.current_anisotropy.to_numpy()[..., None]/10, radius=1, color=0x0000FF)
            self.gui.arrows(orig=self.cells.center.to_numpy()/self.size, direction=self.cells.preferred_major_axis.to_numpy()*self.cells.preferred_anisotropy.to_numpy()[..., None]/10, radius=1, color=0x333333)
        self.gui.show()

    def parse_input(self):
        events = self.gui.get_events()
        for e in events:
            if e.type == ti.GUI.PRESS:

                try:
                    mode = int(e.key)
                    self.display_mode[None] = mode
                    if self.display_mode[None] == 0:
                        print(f"Mode {mode}: Cells ID")
                    elif self.display_mode[None] == 1:
                        print(f"Mode {mode}: Mitosis probabilities")
                    elif self.display_mode[None] == 2:
                        print(f"Mode {mode}: Orientation")
                except ValueError:
                    print(f"Pressed {e.key}")
                    pass

                if e.key == ti.GUI.SPACE:
                    self.sim_pause[None] = 1 - self.sim_pause[None]
                    print(f"Simulation {'paused' if self.sim_pause[None] else 'running'}")


                # If it's left mouse button, print cell info, otherwise select cell

                if e.pos is not None:
                    x, y = e.pos
                    i, j = int(x * self.size), int(y * self.size)
                    cell_id = self.grid[i, j].cell_id
                    
                    if e.key == ti.GUI.LMB: 
                        if cell_id > 0:
                            self.cells[cell_id].print_debug()
                        
                    elif e.key == ti.GUI.RMB:
                        self.cell_selected[None] = cell_id
                        print(f"Selected cell {self.cell_selected[None]} at ({i}, {j})")
                    elif e.key == 's':
                        if cell_id > 0:
                            self.cells[cell_id].should_split = 1
                            print(f"Cell {cell_id} marked for splitting.")
                    

    def run(self):
        while self.gui.running:
            self.parse_input()
            if self.sim_pause[None] == 0:
                self.update()
            self.draw()


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
from multiomicscellsim.taichi_cpm.config.behaviours import EllipticPerimeterConfig, \
                                MitosisAgeVolumeBehaviourConfig



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
            j_adhesion_stroma=0.0,
            j_adhesion_other=4.0,
            preferred_volume_stats=[0.005, 0.0001],
            preferred_anisotropy_stats=[0.8, 0.001],
            preferred_orientation_stats=[[1.0, 0.1], [0.0, 0.1]],
            mitosis_age_stats=[10*10, 50],
            behaviours=[
                            EllipticPerimeterConfig(dynamics=None),
                            MitosisAgeVolumeBehaviourConfig(dynamics=None, 
                                                            sigmoid_slope=0.01, 
                                                            probability_scale=0.1
                                                            )
                       ]
        ),
        CellTypeConfig(
            name="Type 2",
            j_adhesion_stroma=0.0,
            j_adhesion_other=4.0,
            preferred_volume_stats=[0.001, 0.0001],
            preferred_anisotropy_stats=[0.8, 0.001],
            preferred_orientation_stats=[[1.0, 0.1], [0.0, 0.1]],
            mitosis_age_stats=[60*10, 50],
            behaviours=[
                            EllipticPerimeterConfig(dynamics=None),
                            MitosisAgeVolumeBehaviourConfig(dynamics=None, 
                                                            sigmoid_slope=0.01, 
                                                            probability_scale=0.1,
                                                            )
                       ]
        ),
        CellTypeConfig(
            name="Type 3",
            j_adhesion_stroma=0.0,
            j_adhesion_other=4.0,
            preferred_volume_stats=[0.01, 0.0001],
            preferred_anisotropy_stats=[0.2, 0.001],
            preferred_orientation_stats=[[1.0, 0.1], [0.0, 0.1]],
            mitosis_age_stats=[60*10, 50],
            behaviours=[
                            EllipticPerimeterConfig(dynamics=None),
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