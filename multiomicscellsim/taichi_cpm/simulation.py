import taichi as ti
import math
import numpy as np
from perlin_numpy import generate_perlin_noise_2d
from typing import List
from entities import Cell, CellPoint, CellType, MAX_ENERGY_TERMS
from utils import point_in_polygon, get_polygon_edges, local_perimeter
from config.constraints import constraint_factory


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
        self.max_cell_types = sim_config["max_cell_types"]
        self.temperature = sim_config["temperature"]
        self.select_prob = ti.field(dtype=float, shape=())
        self.select_prob[None] = sim_config.get("selection_probability", 0.5) # Probability of selecting a cell for copying

        # Define constraints
        self.constraints = []
        self.constraints_names = []
        
        for constraint_config in sim_config["constraints"]:
            new_constraint = constraint_factory(constraint_config, self)
            self.constraints.append(new_constraint)
            self.constraints_names.append(new_constraint.energy_term_name)

        self.n_constraints = len(self.constraints)

        # Simulation-wide energy coefficients
        self.lambda_chemotaxis = sim_config.get("lambda_chemotaxis", 10)
        self.lambda_orientation = sim_config.get("lambda_orientation", 1.0)
        self.lambda_anisotropy = sim_config.get("lambda_anisotropy", 1.0)
        self.lambda_invasion_penalty = sim_config.get("lambda_invasion_penalty", 0)
        self.lambda_repulsion = sim_config.get("lambda_repulsion", 0)

        # Behavioral parameters
        self.mitosis_probability = sim_config.get("mitosis_probability", 1.0)
        self.mitosis_anisotropy_threshold = sim_config.get("mitosis_anisotropy_threshold", 0.5)

        # Chemicals parameters
        self.chemokine_seed = sim_config.get("chemokine_seed", 0)
        self.chemokine_scale = sim_config.get("chemokine_noise_scale", 6.0)

        # Simulation state
        self.sim_pause = ti.field(dtype=int, shape=())
        self.display_mode = ti.field(dtype=int, shape=())
        self.cell_selected = ti.field(dtype=int, shape=())

        self.gui = ti.GUI(name="MycroVerse", res=self.size)
        self.render_grid = ti.field(dtype=float, shape=(self.size, self.size, 3))
        
        self.build_celltype_list()
        self.reinit()

    def build_celltype_list(self):
        self.cell_types = CellType.field(shape=(len(self.config["cell_types"])+1,))

        for c, ct in enumerate(self.config["cell_types"]):
            # Leave slot 0 for background
            self.cell_types[c+1].j_adhesion_stroma = ct["j_adhesion_stroma"]
            self.cell_types[c+1].j_adhesion_other = ct["j_adhesion_other"]
            self.cell_types[c+1].preferred_volume_stats = ti.Vector(arr=ct["preferred_volume_stats"])
            self.cell_types[c+1].preferred_anisotropy_stats = ti.Vector(arr=ct.get("preferred_anisotropy_stats", [0.0, 0.0]))
            self.cell_types[c+1].preferred_orientation_stats = ti.Vector(arr=ct.get("preferred_orientation_stats", [[0.0, 0.0], [0.0, 0.0]]))
            self.cell_types[c+1].mitosis_age_stats = ti.Vector(arr=ct.get("mitosis_age_stats", [100.0, 10]))

    def reinit(self):
        self.grid = CellPoint.field(shape=(self.size, self.size))
        self.chemokine_grid = ti.field(dtype=float, shape=(self.size, self.size))
        self.cells = Cell.field(shape=(self.max_cells,))
        self.n_cells = ti.field(dtype=int, shape=())
        self.n_cells[None] = 1 # First cell is the background
        self.chemokine_setup()

        # TODO: Stochastic cell generation
        self.create_cell(.5, .5, 1)
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
        # Sample params from stats
        mu_vol, std_vol = self.cell_types[cell_type].preferred_volume_stats
        cell_id = self.n_cells[None]
        self.n_cells[None] += 1
        self.cells[cell_id].cell_type = cell_type
        self.cells[cell_id].preferred_volume = np.clip(self.random.normal(loc=mu_vol, scale=std_vol), .0001, 1) * self.size * self.size
        self.cells[cell_id].preferred_major_axis = ti.Vector(arr=[self.random.normal(loc=self.cell_types[cell_type].preferred_orientation_stats[0, 0],
                                                                                     scale=self.cell_types[cell_type].preferred_orientation_stats[1, 0]),
                                                                  self.random.normal(loc=self.cell_types[cell_type].preferred_orientation_stats[0, 1],
                                                                                     scale=self.cell_types[cell_type].preferred_orientation_stats[1, 1])]).normalized()

        self.cells[cell_id].preferred_anisotropy = self.random.normal(loc=self.cell_types[cell_type].preferred_anisotropy_stats[0],
                                                                                 scale=self.cell_types[cell_type].preferred_anisotropy_stats[1])
        self.cells[cell_id].mitosis_age_threshold = self.random.normal(loc=self.cell_types[cell_type].mitosis_age_stats[0],
                                                                       scale=self.cell_types[cell_type].mitosis_age_stats[1])
        print(f"Creating cell {cell_id} of type {cell_type} with preferred volume {self.cells[cell_id].preferred_volume}, \
                orientation {self.cells[cell_id].preferred_major_axis}, \
                anisotropy {self.cells[cell_id].preferred_anisotropy}, \
                mitosis age threshold {self.cells[cell_id].mitosis_age_threshold}")
        
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
                self.cells[c].current_perimeter = 0.0
                self.cells[c].current_volume = 0.0
                self.cells[c].center = ti.Vector([0.0, 0.0])
                self.cells[c].covariance_matrix = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                self.cells[c].current_anisotropy = 0.0
                self.cells[c].maj_axis = ti.Vector([0.0, 0.0])
                self.cells[c].min_axis = ti.Vector([0.0, 0.0])


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
                self.cells[c].maj_axis, self.cells[c].min_axis, self.cells[c].max_eigenvalue, self.cells[c].min_eigenvalue = self.compute_eigenvectors_and_eigenvalues(self.cells[c].covariance_matrix)
                self.cells[c].current_anisotropy = (self.cells[c].max_eigenvalue - self.cells[c].min_eigenvalue) / (self.cells[c].max_eigenvalue + self.cells[c].min_eigenvalue + 1e-6)

            # Update behavior
            if c > 0 and self.cells[c].cell_type > 0:
                # Try to approximate an ellipse
                # TODO: If more shape are implemented, this should be generalized

                a = ti.sqrt(self.cells[c].max_eigenvalue)
                b = ti.sqrt(self.cells[c].min_eigenvalue)

                scaling_factor = ti.sqrt(self.cells[c].current_volume / (ti.math.pi * a * b))
                a *= scaling_factor
                b *= scaling_factor

                # Step 3: Ramanujan's perimeter approximation
                h = ((a - b)**2) / ((a + b)**2)
                self.cells[c].preferred_perimeter = 3 * ti.math.pi * (a + b) * (1 + (3 * h) / (10 + ti.sqrt(4 - 3 * h)))

                # Recompute energy terms
                # TODO: Remove after implementing all constraints
                # FIXME: Find a solution for this. We cannot iterate over constraints in Taichi
                for cid in ti.static(range(self.n_constraints)):
                    self.cells[c].current_energy_terms[self.constraints[cid].energy_index] = self.constraints[cid].calculate_current_cell_energy(c)

                self.cells[c].current_anisotropy_energy = self.lambda_anisotropy * (self.cells[c].current_anisotropy - self.cells[c].preferred_anisotropy) ** 2
                self.cells[c].current_orientation_energy = self.lambda_orientation * (1.0 - self.cells[c].maj_axis.dot(self.cells[c].preferred_major_axis.normalized())) ** 2

                # Compute mitosis probabilities
                k = 0.01 # Slope of sigmoid
                self.cells[c].mitosis_prob_volume = (1.0 / (1.0 + ti.exp(-k * (self.cells[c].current_volume - self.cells[c].preferred_volume))))
                self.cells[c].mitosis_prob_age = (1.0 / (1.0 + ti.exp(-k * (self.cells[c].current_age - self.cells[c].mitosis_age_threshold))))


    @ti.kernel
    def do_copy(self):

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
        
        for cell_id in range(self.n_cells[None]):
            if cell_id > 0 and self.cells[cell_id].should_split > 0:
                new_cell_id = self.n_cells[None]
                
                if new_cell_id < self.max_cells:    
                    # Split from center to a random direction
                    old_center = self.cells[cell_id].center
                    #split_direction = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]).normalized()
                    split_direction = self.cells[cell_id].maj_axis.normalized() # Use the major axis as the split direction
                    #print(f"Cell {cell_id} Splitting. Orientation: {self.cells[cell_id].maj_axis}, Direction: {split_direction}, Center: {old_center}")
                    #split_direction = self.cells[cell_id].current_polarization.normalized()
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
                    #print(f"Cell {cell_id}: Splitted. Size {self.cells[cell_id].current_volume} new cell {new_cell_id} has {new_current_size} pixels")
                    
                    # TODO: Here we could implement mutations and stochasticity
                    self.cells[new_cell_id].cell_type = self.cells[cell_id].cell_type
                    self.cells[new_cell_id].preferred_volume = self.cells[cell_id].preferred_volume            
                    self.cells[new_cell_id].preferred_perimeter = self.cells[cell_id].preferred_perimeter
                    self.cells[new_cell_id].preferred_major_axis = self.cells[cell_id].preferred_major_axis
                    self.cells[new_cell_id].preferred_anisotropy = self.cells[cell_id].preferred_anisotropy
                    self.cells[new_cell_id].mitosis_age_threshold = self.cells[cell_id].mitosis_age_threshold
                    self.cells[new_cell_id].should_split = 0
                    self.cells[new_cell_id].current_age = 0.0
                    # FIXME: Is this correct?
                    self.cells[cell_id].current_age = 0.0



    @ti.func
    def compute_eigenvectors_and_eigenvalues(self, mat: ti.math.mat2) -> ti.math.vec2: # type: ignore
        eigvals, eigvects = ti.sym_eig(mat)
        # Compare the eigenvalues explicitly
        # Depending on their order, select correctly
        cond = eigvals[0] > eigvals[1]
        longest_axis = ti.select(cond, eigvects[:, 0], eigvects[:, 1])
        shortest_axis = ti.select(cond, eigvects[:,  1], eigvects[:, 0])
        longest_eigenvalue = ti.select(cond, eigvals[0], eigvals[1])
        shortest_eigenvalue = ti.select(cond, eigvals[1], eigvals[0])

        return longest_axis, shortest_axis, longest_eigenvalue, shortest_eigenvalue

   
    @ti.kernel
    def select_potential_copies(self):
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
    
    @ti.func
    def calc_adhesion(self, i:int, j:int, cell_id:int) -> float:
        """
            Calculate the local adhesion of a pixel considering its cell_id as the given value
            so it can be used to simulate copies.
        """
        energy = 0.0
        for i_offset in range(-1, 2):
            for j_offset in range(-1, 2):
                if not self.is_out_of_bounds(i+i_offset, j+j_offset) and \
                   cell_id != self.grid[i+i_offset, j+j_offset].cell_id:
                   if self.grid[i+i_offset, j+j_offset].cell_id == 0:
                       ti.atomic_add(energy, self.cell_types[self.cells[cell_id].cell_type].j_adhesion_stroma)
                   else:
                       ti.atomic_add(energy, self.cell_types[self.cells[cell_id].cell_type].j_adhesion_other)
        return energy

    @ti.func
    def calc_orientation_anisotropy_h(self, s_i:int, s_j:int, t_i:int, t_j:int) -> float:
        total = 0.0

        if s_i == t_i and s_j == t_j:
            # "Current" energy of the two relevant cells
            for c in range(self.n_cells[None]):
                if c > 0 and (c == self.grid[s_i, s_j].cell_id or c == self.grid[t_i, t_j].cell_id):
                    A = self.cells[c].current_anisotropy
                    A_star = self.cells[c].preferred_anisotropy
                    total += self.lambda_anisotropy * (A - A_star) ** 2
                    if self.lambda_orientation > 0.0:
                        # Use current major axis and preferred_major_axis
                        emax = self.cells[c].maj_axis
                        ustar = self.cells[c].preferred_major_axis.normalized()
                        align2 = ti.min(1.0, ti.max(-1.0, emax.dot(ustar))) ** 2
                        total += self.lambda_orientation * (1.0 - align2)
        else:
            # After-copy energy (rank-1 update of center & covariance), for src & tgt only
            t_pos = ti.Vector([float(t_i), float(t_j)])
            for c in range(self.n_cells[None]):
                if c > 0 and (c == self.grid[s_i, s_j].cell_id or c == self.grid[t_i, t_j].cell_id):
                    N = self.cells[c].current_volume
                    gain = 1 if (c == self.grid[s_i, s_j].cell_id) else -1 if (c == self.grid[t_i, t_j].cell_id) else 0
                    newN = N + gain
                    # guard tiny cells
                    newN = ti.max(1.0, newN)

                    new_center = ti.Vector([0.0, 0.0])
                    # center update
                    if gain == 1:
                        new_center = (N * self.cells[c].center + t_pos) / newN
                    elif gain == -1:
                        new_center = (N * self.cells[c].center - t_pos) / newN
                    else:
                        new_center = self.cells[c].center

                    # covariance update (about new center); same pattern you already use
                    diff = new_center - t_pos
                    new_cov = (N * self.cells[c].covariance_matrix + gain * diff.outer_product(diff)) / newN

                    # eig + anisotropy
                    emax, emin, eva_max, eva_min = self.compute_eigenvectors_and_eigenvalues(new_cov)
                    A_after = (eva_max - eva_min) / (eva_max + eva_min + 1e-6)
                    A_star = self.cells[c].preferred_anisotropy
                    E = self.lambda_anisotropy * (A_after - A_star) ** 2

                    if self.lambda_orientation > 0.0:
                        ustar = self.cells[c].preferred_major_axis.normalized()
                        align2 = ti.min(1.0, ti.max(-1.0, emax.dot(ustar))) ** 2
                        E += self.lambda_orientation * (1.0 - align2)

                    total += E

        return total


    @ti.kernel
    def calc_energy(self):
        """
            Calculate the energy of the potential copies.
            For each selected source pixel, calculate the energy gain of copying it into its target pixel.
            The energy is calculated as:
                - Adhesion delta: adhesion of source + adhesion of target - adhesion of target after copy - adhesion of source after copy
                - Volume delta: Hvol after copy - Hvol before copy
                - Perimeter delta: H_per after copy - H_per before copy
        """

        for i, j in self.grid:
            if self.grid[i, j].selected_as_src:
               
                t_i, t_j = ti.cast(self.grid[i, j].copy_into[0], int), ti.cast(self.grid[i, j].copy_into[1], int)
                # Adhesion delta is calculated locally by summing adhesion of source and target pixels
                # After Copy - Before copy
                adhesion_after = self.calc_adhesion(t_i, t_j, self.grid[i, j].cell_id) + self.calc_adhesion(i, j, self.grid[i, j].cell_id)
                adhesion_current = self.calc_adhesion(t_i, t_j, self.grid[t_i, t_j].cell_id) + self.calc_adhesion(i, j, self.grid[i, j].cell_id)
                delta_adhesion = adhesion_after - adhesion_current
                ti.atomic_add(self.grid[i, j].copy_energy_delta, delta_adhesion)
 
                # FIXME: Find a solution for this. We cannot iterate over constraints in Taichi
                for cid in ti.static(range(self.n_constraints)):
                    ti.atomic_add(self.grid[i, j].copy_energy_delta, self.constraints[cid].calculate_energy_delta(s_i=i, s_j=j, t_i=t_i, t_j=t_j))
                  
                # Polarization Bias:
                # This is a bias term that encourages cells to copy in the direction of their preferred major axis
                #polarization_bias_h = self.calc_polarization_bias_h(s_i=i, s_j=j, t_i=t_i, t_j=t_j)
                #ti.atomic_add(self.grid[i, j].copy_energy_delta, polarization_bias_h)

                # Chemotaxis (bias term):
                chemotaxis_h = -self.lambda_chemotaxis*(self.chemokine_grid[t_i, t_j] - self.chemokine_grid[i, j]) - self.lambda_repulsion*self.calc_chemotaxis_repulsion_h(s_i=i, s_j=j, t_i=t_i, t_j=t_j)
                ti.atomic_add(self.grid[i, j].copy_energy_delta, chemotaxis_h)
                
                orientation_anisotropy_before = self.cells[self.grid[i, j].cell_id].current_anisotropy_energy + \
                                                self.cells[self.grid[t_i, t_j].cell_id].current_anisotropy_energy + \
                                                self.cells[self.grid[i, j].cell_id].current_orientation_energy + \
                                                self.cells[self.grid[t_i, t_j].cell_id].current_orientation_energy
                

                orientation_anisotropy_after = self.calc_orientation_anisotropy_h(s_i=i, s_j=j, t_i=t_i, t_j=t_j)
                delta_orientation_anisotropy = orientation_anisotropy_after - orientation_anisotropy_before
                ti.atomic_add(self.grid[i, j].copy_energy_delta, delta_orientation_anisotropy)
 
                # Avoid invading other cells
                if self.grid[i, j].cell_id > 0 and self.grid[t_i, t_j].cell_id > 0:
                    ti.atomic_add(self.grid[i, j].copy_energy_delta, self.lambda_invasion_penalty)
 
 
                # If the cell is selected and either source or target, print debug info
                #print(f"Selected Cell: {self.grid[i, j].cell_id} ({i}, {j}) -> ({t_i}, {t_j}). {self.cell_selected}")
                if self.cell_selected[None] > 0 and (self.cell_selected[None] == self.grid[i, j].cell_id or self.cell_selected[None] == self.grid[t_i, t_j].cell_id):
                   print(chr(27) + "[2J")
                   print(f"Cell {self.cell_selected[None]} Selected. Copying from {i}, {j} to {t_i}, {t_j}")
                   print(f"Copy from {i}, {j} ({self.grid[i, j].cell_id}) to {t_i}, {t_j} ({self.grid[t_i, t_j].cell_id})")
                   print(f"Adhesion Energy (Current/After): {adhesion_current} / {adhesion_after} = {delta_adhesion}")
                   #print(f"Volume Energy (Current/After): {delta_volume_current} / {delta_volume_after} = {delta_volume}")
                   #print(f"Perimeter Energy: {delta_perimeter}")
                   print(f"Orientation/Anisotropy Energy (Current/After): {orientation_anisotropy_before} / {orientation_anisotropy_after} = {delta_orientation_anisotropy}")
                   print(f"Chemotaxis Energy: {chemotaxis_h}")

    @ti.func
    def calc_polarization_bias_h(self, s_i: int, s_j: int, t_i: int, t_j: int) -> float:
        bias = 0.0  # Default bias

        src_cell_id = self.grid[s_i, s_j].cell_id

        # If source pixel is background (cell_id == 0), bias stays zero
        if src_cell_id != 0:
            pref_pol = self.cells[src_cell_id].preferred_major_axis.normalized()
            copy_dir = ti.Vector([t_i - s_i, t_j - s_j], dt=ti.f32)
            norm_copy_dir = copy_dir.norm()

            if norm_copy_dir > 1e-6:
                copy_dir = copy_dir / norm_copy_dir
                dot = pref_pol.dot(copy_dir)
                bias = -self.lambda_polarization * dot * self.cells[src_cell_id].current_volume

        return bias
            
    @ti.func
    def calc_chemotaxis_repulsion_h(self, s_i:int, s_j:int, t_i:int, t_j:int) -> float:
        # Follow chemokine gradient
        copy_direction = (ti.Vector([t_i, t_j]) - ti.Vector([s_i, s_j])).normalized()
        
        desired_direction = ti.Vector([0.0, 0.0])
        power = 0.0

        source_type = self.cells[self.grid[s_i, s_j].cell_id].cell_type
        target_type = self.cells[self.grid[t_i, t_j].cell_id].cell_type

        if source_type != target_type:
            if source_type == 0:
                # Source is background, energy is governed by target
                closest_center = ti.Vector([float("inf"), float("inf")])
                center = self.cells[self.grid[t_i, t_j].cell_id].center
                for c in range(self.n_cells[None]):
                    if c > 0 and c != self.grid[t_i, t_j].cell_id:
                        dist = (self.cells[c].center - center).norm()
                        if dist < (closest_center - center).norm():
                            closest_center = self.cells[c].center
                desired_direction = -(closest_center - center).normalized()
                power = 1.0
            else:
                # Source is a cell, energy is governed by source
                closest_center = ti.Vector([float("inf"), float("inf")])
                center = self.cells[self.grid[s_i, s_j].cell_id].center
                for c in range(self.n_cells[None]):
                    if c > 0 and c != self.grid[s_i, s_j].cell_id:
                        dist = (self.cells[c].center - center).norm()
                        if dist < (closest_center - center).norm():
                            closest_center = self.cells[c].center
                desired_direction = -(closest_center - center).normalized()
                power = 1.0
        dot = 0.0
        if desired_direction.norm() > 1e-6:
            desired_dir_norm = desired_direction.normalized()
            dot = desired_dir_norm.dot(copy_direction)
        # Calculate the angle between the desired direction and the copy direction
        #print(f"Desired Direction: {desired_direction}, Copy Direction: {copy_direction} -> Dot: {dot}")
        return dot*power


    @ti.func
    def calc_polarization_h(self, s_i: int, s_j: int, t_i:int, t_j: int) -> float:
        """
            Calculate the polarization energy of a pixel assuming it is copied from s to t.
            The energy is calculated as the difference between the preferred polarization and the current polarization
        """ 
        
        t_pos = ti.Vector([float(t_i), float(t_j)])
        total_energy = 0.0
        N = int(0)
        new_N = int(0)
        gain = int(0)

        # If the source and target are the same, we just compute the current polarization energy

        if s_i == t_i and s_j == t_j:
            # If source and target are the same, we just compute the current polarization energy
            for c in range(self.n_cells[None]):
                if c > 0:
                    if (c == self.grid[s_i, s_j].cell_id or c == self.grid[t_i, t_j].cell_id):
                        N = self.cells[c].current_volume
                        new_N = N
                        
                        alignment = abs(self.cells[c].current_polarization.dot(self.cells[c].preferred_polarization))

                        #print(f"Cell {c} New Polarization: {new_polarization_vector}, Preferred: {preferred_polarization}, Alignment: {alignment}")
                        # TODO: Polarization strength
                        # TODO: When called with same source and target, should return the current polarization energy
                        ti.atomic_add(total_energy, self.lambda_polarization * (1 - alignment)**2)
                        #ti.atomic_add(total_energy, 100*self.lambda_polarization * (1.0 - self.cells[c].current_anisotropy)**2)
        else:
             for c in range(self.n_cells[None]):
                gain = 0
                new_N = 0
                # All cells that are not source or target will keep the same polarization so they are not considered
                if c > 0:
                    if (c == self.grid[s_i, s_j].cell_id or c == self.grid[t_i, t_j].cell_id):
                        N = self.cells[c].current_volume
                        if (c == self.grid[s_i, s_j].cell_id):
                            # Source is gaining a pixel
                            gain = int(1)
                        if (c == self.grid[t_i, t_j].cell_id):
                            # Target is losing a pixel
                            gain = int(-1)

                        new_N = self.cells[c].current_volume + gain
                        # To compute the polarization energy after copy, we need to compute to consider that a gain/loss of pixel changes:
                        # 1. The center of mass
                        new_center = ti.Vector([0.0, 0.0])
                        
                        if (c == self.grid[s_i, s_j].cell_id):
                            new_center = (N * self.cells[c].center + t_pos) / new_N
                        elif (c == self.grid[t_i, t_j].cell_id):
                            new_center = (N * self.cells[c].center - t_pos) / new_N
                        
                        # 2. The covariance matrix
                        # Instead of recomputing the covariance matrix, we can use the current one and adjust it based on the new center of mass
                        # I.e., we assume the old pixels are still there, but the new pixel changes the center of mass and the covariance matrix
                        new_covariance_matrix = (1 / new_N) * (N * self.cells[c].covariance_matrix + gain*(new_center - t_pos).outer_product(new_center - t_pos))
                        # 3. The polarization vector (longest axis of the covariance matrix)
                        new_polarization_vector, _, new_anisotropy = self.compute_polarization_and_anisotropy(new_covariance_matrix)
                        
                        alignment = abs(new_polarization_vector.dot(self.cells[c].preferred_polarization))
                        #print(f"Cell {c} New Polarization: {new_polarization_vector}, Preferred: {preferred_polarization}, Alignment: {alignment}")
                        # TODO: Polarization strength
                        ti.atomic_add(total_energy, self.lambda_polarization * (1.0 - alignment)**2)
                        #ti.atomic_add(total_energy, 100*self.lambda_polarization * (1.0 - new_anisotropy)**2)


        return total_energy

    def cpm_step(self):
        # Sources and targets are stored into .selected* and .copy_*
        self.select_potential_copies()
        # Avoiding race conditions (where multiple pixels wants to source/target the same pixels)
        self.calc_energy()
        self.do_copy()

        # Update cell grid parameters from grid:
        self.update_grid_params()
    
    @ti.kernel
    def check_mitosis(self):
        """
            Check if any cell should split based on its current volume.
            If the current volume is greater than the preferred volume, mark it for splitting.
        """
        for c in self.cells:
            if self.cells[c].cell_type >= 1:
               
                #print(f"Vol Prob for cell {c}: {vol_prob} (Current Volume: {v}, Preferred Volume: {v0})")
                if ti.random() < self.mitosis_probability * self.cells[c].mitosis_prob_volume * self.cells[c].mitosis_prob_age:
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
                        self.render_grid[i, j, 0] = self.cells[self.grid[i, j].cell_id].mitosis_prob_volume
                        self.render_grid[i, j, 1] = self.cells[self.grid[i, j].cell_id].mitosis_prob_age
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


from config.constraints import PerimeterConstraintConfig, VolumeConstraintConfig

sim_config = {
    "random_seed": 42,
    "size": 512,
    "max_cell_types": 10,
    "max_cells": 100000,
    "temperature": 1,
    "selection_probability": 1,
    "lambda_volume": 1e-3,
    "lambda_perimeter": 1e-3,
    "lambda_chemotaxis": 50,
    "lambda_invasion_penalty": 1,
    "lambda_orientation": 0,
    "lambda_anisotropy": 0,
    "lambda_repulsion": 0,
    "mitosis_anisotropy_threshold": 0.7,
    "mitosis_probability": 0.2,
    "chemokine_seed": 0,
    "chemokine_noise_scale": 7,
    "constraints": [
        VolumeConstraintConfig(lambda_weight=1e-3, energy_index=0),
        PerimeterConstraintConfig(lambda_weight=1e-3, energy_index=1)
    ],
    "cell_types": [
        {
            "j_adhesion_stroma": 0.0,
            "j_adhesion_other": .0,
            "preferred_volume_stats": [0.005, 0.0001],
            "preferred_anisotropy_stats": [0.9, 0.001],
            "preferred_orientation_stats": [[1.0, 0.1], [0.0, 0.1]],
            "mitosis_age_stats": [60*10, 50]
        }
    ]
}

sim = Simulation(sim_config)
sim.run()