import taichi as ti
import math
import numpy as np
from typing import List

ti.init(arch=ti.cpu, debug=False)


@ti.dataclass
class CellType():
    j_adhesion_stroma: float
    j_adhesion_other: float
    preferred_volume_stats: ti.math.vec2 # mean and std for preferred volume
    
@ti.dataclass
class Cell():
    cell_id: int
    cell_type: int
    preferred_volume: float
    preferred_perimeter: float
    current_volume: float
    current_perimeter: float
    center: ti.math.vec2
    should_split: int

@ti.dataclass
class CellPoint():
    cell_id: int
    cell_type: int
    is_membrane: int # 1 if it is a membrane (has neighbors with different cell_id)
    local_perimeter: int # number of neighbors with different cell_id
    selected_as_src: int
    selected_as_trg: int
    copy_into: ti.math.vec2
    copy_from: ti.math.vec2
    copy_energy_delta: float


@ti.data_oriented
class Simulation():
    def __init__(self, sim_config):
        self.config = sim_config
        self.size = sim_config["size"]
        self.max_cells=sim_config["max_cells"]
        self.max_cell_types = sim_config["max_cell_types"]
        self.lambda_volume = sim_config["lambda_volume"]
        self.lambda_perimeter = sim_config["lambda_perimeter"]
        self.gui = ti.GUI(name="MycroVerse", res=self.size)
        self.render_grid = ti.field(dtype=float, shape=(self.size, self.size, 3))
        self.select_prob = .5
        self.build_celltype_list()
        self.reinit()

    def build_celltype_list(self):
        self.cell_types = CellType.field(shape=(len(self.config["cell_types"])+1,))

        for c, ct in enumerate(self.config["cell_types"]):
            # Leave slot 0 for background
            self.cell_types[c+1].j_adhesion_stroma = ct["j_adhesion_stroma"]
            self.cell_types[c+1].j_adhesion_other = ct["j_adhesion_other"]
            self.cell_types[c+1].preferred_volume_stats = ti.Vector(arr=ct["preferred_volume_stats"])

    def reinit(self):
        self.grid = CellPoint.field(shape=(self.size, self.size))
        self.cells = Cell.field(shape=(self.max_cells,))
        self.n_cells = ti.field(dtype=int, shape=())
        self.n_cells[None] = 1 # First cell is the background
        self.create_cell(.2, .2, 1)
        self.create_cell(.7, .7, 1)
        # Recalc all params before starting simulation
        self.reset_grid_params()
    
    @ti.func
    def point_in_polygon(self, x:int, y:int, polygon: ti.template()) -> int: # type: ignore
        """ 
            Ray-casting algorithm to check if a point is inside a polygon 
            Args:
                x, y: coordinates to test
                polygon: a 2D vector field containing the edges of the polygon
        """
        count = 0
        n_verts = polygon.shape[0]
        for i in range(n_verts):
            a, b = polygon[i], polygon[(i+1) % n_verts]
            if (a.y > y) != (b.y > y):  # Edge crosses the horizontal line at y
                slope = (b.x - a.x) / (b.y - a.y)
                intersect_x = a.x + slope * (y - a.y)
                if x < intersect_x:  # Count only if intersection is to the right
                    count += 1
        return ti.cast(count % 2 == 1, int)

    @ti.kernel
    def get_polygon_edges(self, xc: float, yc:float, n_edges: int, radius: float, verts: ti.template()): # type: ignore
        """
            Update verts to contain the edges of a polygon
        """
        angle_step = 2 * math.pi / n_edges
        for i in range(n_edges):
            angle = i * angle_step
            c = ti.cast((xc + radius * ti.cos(angle))*self.size, int)
            r = ti.cast((yc + radius * ti.sin(angle))*self.size, int)
            verts[i] = [r, c]

    @ti.kernel
    def draw_polygon_on_grid(self, polygon:ti.template(), cell_id:int, cell_type:int): # type: ignore
        for i, j in self.grid:
            if self.point_in_polygon(i, j, polygon=polygon):
                self.grid[i,j].cell_id = cell_id
                self.grid[i,j].cell_type = cell_type


    def create_cell(self, xc: float, yc: float, cell_type:int):
        
        # Sample params from stats
        mu_vol, std_vol = self.cell_types[cell_type].preferred_volume_stats
        cell_id = self.n_cells[None]
        self.n_cells[None] += 1
        self.cells[cell_id].cell_type = cell_type
        self.cells[cell_id].preferred_volume = np.clip(np.random.normal(loc=mu_vol, scale=std_vol), .0001, 1) * self.size * self.size

        radius = 0.02
        n_edges = 16
        verts = ti.Vector.field(n=2, dtype=int, shape=(n_edges,))
        self.get_polygon_edges(xc, yc, n_edges=n_edges, radius=radius, verts=verts)
        self.draw_polygon_on_grid(polygon=verts, cell_id=cell_id, cell_type=cell_type)
        

    @ti.func
    def is_out_of_bounds(self, i:int, j:int) -> int:
        return ti.cast(i< 0 or i >= self.size or j<0 or j>=self.size, int)

    @ti.func
    def local_perimeter(self, i, j, grid_value) -> int:
        """
            Returns the local perimeter of a pixel in the grid assuming the given grid_value as cell_id.
            (This allows to calculate the perimeter of a pixel assuming it is part of a different cell)
        """
        local_perimeter = 0 
        for i_offset in range(-1, 2):
            for j_offset in range(-1, 2):
                if grid_value > 0 and \
                   not self.is_out_of_bounds(i+i_offset, j+j_offset) and \
                   grid_value != self.grid[i+i_offset, j+j_offset].cell_id:
                   ti.atomic_add(local_perimeter, 1)
        return local_perimeter

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
    def reset_grid_params(self):
        for c in self.cells:
            if self.cells[c].cell_type > 0:
                self.cells[c].current_perimeter = 0.0
                self.cells[c].current_volume = 0.0
                self.cells[c].center = ti.Vector([0.0, 0.0])
                self.cells[c].should_split = 0
                
            
        for i, j in self.grid:
            self.grid[i, j].local_perimeter = self.local_perimeter(i, j, self.grid[i, j].cell_id)
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
                self.cells[c].center /= self.cells[c].current_volume
                #print(f"Cell {c} Center: {self.cells[c].center[0]} {self.cells[c].center[1]}")
                # Recompute preferred perimeter to keep roundness based on current volume
                self.cells[c].preferred_perimeter = 2 * math.pi * ti.sqrt(self.cells[c].preferred_volume / math.pi)
                

    @ti.kernel
    def do_copy(self):
        #TODO: Implement boltzmann
        for i, j in self.grid:
            if self.grid[i, j].selected_as_src:
                t_i, t_j = ti.cast(self.grid[i, j].copy_into[0], int), ti.cast(self.grid[i, j].copy_into[1], int)
                if self.grid[i, j].copy_energy_delta < 0:
                    self.grid[t_i, t_j].cell_type = self.grid[i, j].cell_type
                    self.grid[t_i, t_j].cell_id = self.grid[i, j].cell_id
                    self.grid[t_i, t_j].selected_as_src = 0
                    self.grid[t_i, t_j].selected_as_trg = 0
                    self.grid[t_i, t_j].copy_energy_delta = 0

    @ti.kernel
    def trigger_mitosis(self):
        
        for cell_id in range(self.n_cells[None]):
            if cell_id > 0 and self.cells[cell_id].should_split > 0:
                new_cell_id = self.n_cells[None]
                
                if new_cell_id < self.max_cells:    
                    # Split from center to a random direction
                    old_center = self.cells[cell_id].center
                    split_direction = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]).normalized()
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
                    
                    ti.atomic_add(self.n_cells[None], 1)
                    print(f"Cell {cell_id}: Splitted. Size {self.cells[cell_id].current_volume} new cell {new_cell_id} has {new_current_size} pixels")
                    self.cells[new_cell_id].cell_type = self.cells[cell_id].cell_type
                    self.cells[new_cell_id].preferred_volume = self.cells[cell_id].preferred_volume                

    @ti.kernel
    def select_potential_copies(self):
        for i, j in self.grid:
            if ti.random() < self.select_prob and \
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
    def calc_volume_h(self, src_cell_id: int, tgt_cell_id: int) -> float:
        total_energy = 0.0
        # Taichi does not support nested for...
        for c in range(self.n_cells[None]):
            gain = 0.0
            if src_cell_id == c:
                # Current Volume gain one pixel
                gain += 1.0
            if tgt_cell_id == c:
                # Current Volume loses one pixel
                gain -= 1.0
            total_energy += self.lambda_volume * (self.cells[c].current_volume + gain - self.cells[c].preferred_volume)**2
        return total_energy

    @ti.func
    def calc_perimeter_h(self, s_i:int, s_j:int, t_i:int, t_j:int, t_value:int) -> float:
        """
            Calculate the perimeter energy of a pixel assuming it is copied from s to t.
            To calculate the current perimeter, pass the same i,j as s_i, s_j and t_i, t_j
        """
        total_energy = 0.0

        for c in range(self.n_cells[None]):
            # All cells that are not source or target will keep the same perimeter so they are not considered
            
            if c > 0 and (c == t_value or c == self.grid[s_i, s_j].cell_id):
                gain_perimeter = 0.0
                current_perimeter = self.cells[c].current_perimeter

                # If the source and target are the same, we have no gain, otherwise...
                if s_i != t_i or s_j != t_j:
                    # We just consider the neighborhood of the target pixel (which is the only one that changes)
                    same_cell_neighbors = 8 - self.local_perimeter(t_i, t_j, c)
                    if c == t_value:
                        gain_perimeter = -8 + 2*same_cell_neighbors
                    else:
                        gain_perimeter = 8 - 2*same_cell_neighbors
            
                ti.atomic_add(total_energy, self.lambda_perimeter * (current_perimeter + gain_perimeter - self.cells[c].current_volume)**2)

        return total_energy

        

    @ti.kernel
    def calc_energy(self):
        for i, j in self.grid:
            if self.grid[i, j].selected_as_src:
               t_i, t_j = ti.cast(self.grid[i, j].copy_into[0], int), ti.cast(self.grid[i, j].copy_into[1], int)
               # Adhesion delta is calculated locally by summing adhesion of source and target pixels
               # After Copy - Before copy
               ti.atomic_add(self.grid[i, j].copy_energy_delta,
                    (self.calc_adhesion(t_i, t_j, self.grid[i, j].cell_id) + \
                     self.calc_adhesion(i, j, self.grid[i, j].cell_id)) - \
                    (self.calc_adhesion(t_i, t_j, self.grid[t_i, t_j].cell_id) + \
                     self.calc_adhesion(i, j, self.grid[i, j].cell_id)))

               # Volume: Hvol after copy - Current Hvol (0 gain given by src==target)
               delta_volume = self.calc_volume_h(src_cell_id=self.grid[i, j].cell_id, tgt_cell_id=self.grid[t_i, t_j].cell_id) - self.calc_volume_h(src_cell_id=self.grid[i, j].cell_id, tgt_cell_id=self.grid[i, j].cell_id)
               ti.atomic_add(self.grid[i, j].copy_energy_delta, delta_volume)

               # Perimeter:
               # delta_perimter = H_per after copy - Current H_per
               delta_perimeter = self.calc_perimeter_h(i, j, t_i, t_j, self.grid[t_i, t_j].cell_id) - self.calc_perimeter_h(i, j, i, j, self.grid[i, j].cell_id)
               ti.atomic_add(self.grid[i, j].copy_energy_delta, delta_perimeter)


    def cpm_step(self):
        # Sources and targets are stored into .selected* and .copy_*
        self.select_potential_copies()
        # Avoiding race conditions (where multiple pixels wants to source/target the same pixels)
        self.calc_energy()
        self.do_copy()

        # Update cell grid parameters from grid:
        self.reset_grid_params()
    
    @ti.kernel
    def check_mitosis(self):
        for c in self.cells:
            if self.cells[c].current_volume > self.cells[c].preferred_volume:
                self.cells[c].should_split = 1

    def biology_events(self):
        self.check_mitosis()
        self.trigger_mitosis()
        self.reset_grid_params()


    def update(self):
        self.cpm_step()
        self.biology_events()
    
    @ti.kernel
    def render(self):
        for i, j in self.grid:
            self.render_grid[i, j, 0] = self.grid[i, j].cell_id / self.n_cells[None]
            self.render_grid[i, j, 1] = self.grid[i, j].local_perimeter / 8
            self.render_grid[i, j, 2] = 0.0

    def draw(self):
        self.gui.clear()
        self.render_grid.fill(0)
        self.render()
        self.gui.set_image(self.render_grid)
        self.gui.show()

    def parse_input(self):
        events = self.gui.get_events()
        pass

    def run(self):
        while self.gui.running:
            self.parse_input()
            self.update()
            self.draw()



sim_config = {
    "size": 512,
    "max_cell_types": 10,
    "max_cells": 1000,
    "lambda_volume": 1,
    "lambda_perimeter": 0.5,
    "cell_types": [
        {
            "j_adhesion_stroma": 0.01*8,
            "j_adhesion_other": .1*8,
            "preferred_volume_stats": [0.005, 0.001],
         }
    ]
}

sim = Simulation(sim_config)
sim.run()