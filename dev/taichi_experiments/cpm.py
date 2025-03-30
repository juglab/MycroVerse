import taichi as ti
import math
import numpy as np
from typing import List

ti.init(arch=ti.cpu, debug=False)

# TODO: Make also self.grid sparse

@ti.dataclass
class CellPoint():
    cell_id: int
    cell_type: int
    is_membrane: int
    selected_as_src: int
    selected_as_trg: int
    copy_into: ti.math.vec2
    copy_from: ti.math.vec2


@ti.data_oriented
class Simulation():
    def __init__(self, size=512, headless=False):
        
        self.size = size
        self.gui = ti.GUI(name="MycroVerse", res=self.size)
        self.render_grid = ti.field(dtype=float, shape=(self.size, self.size, 3))
        self.select_prob = 1.0
        self.reinit()

    def reinit(self):
        self.grid = CellPoint.field(shape=(self.size, self.size))
        self.draw_cell(.5, .5, .2, 1, 1)
    
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


    def draw_cell(self, xc: float, yc: float, radius: float, cell_id:int, cell_type:int):
        
        n_edges = 16
        verts = ti.Vector.field(n=2, dtype=int, shape=(n_edges,))
        self.get_polygon_edges(xc, yc, n_edges=n_edges, radius=radius, verts=verts)
        self.draw_polygon_on_grid(polygon=verts, cell_id=cell_id, cell_type=cell_type)     

    @ti.func
    def is_out_of_bounds(self, i:int, j:int) -> int:
        return ti.cast(i< 0 or i >= self.size or j<0 or j>self.size, int)

    @ti.func
    def is_membrane(self, i, j):
        membrane = 0 
        for i_offset in range(-1, 2):
            for j_offset in range(-1, 2):
                if self.grid[i, j].cell_id > 0 and \
                not self.is_out_of_bounds(i+i_offset, j+j_offset) and \
                self.grid[i, j].cell_id != self.grid[i+i_offset, j+j_offset].cell_id:
                    membrane = 1
        return membrane

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
                ti.atomic_add(neigh[0], oi)
                ti.atomic_add(neigh[1], oj)
        return neigh
            
    @ti.kernel
    def find_membranes(self):
        for i, j in self.grid:
            self.grid[i, j].is_membrane = self.is_membrane(i, j)

    @ti.kernel
    def reset_grid_temps(self):
        for i, j in self.grid:
            self.grid[i, j].is_membrane = self.is_membrane(i, j)
            self.grid[i, j].selected_as_src = 0
            self.grid[i, j].selected_as_trg = 0
            self.grid[i, j].copy_into.fill(0)
            self.grid[i, j].copy_from.fill(0)

    @ti.kernel
    def do_copy(self):
        for i, j in self.grid:
            if self.grid[i, j].selected_as_trg:
                s_i, s_j = ti.cast(self.grid[i, j].copy_from[0], int), ti.cast(self.grid[i, j].copy_from[1], int)
                self.grid[i, j].cell_type = self.grid[s_i, s_j].cell_type
                self.grid[i, j].cell_id = self.grid[s_i, s_j].cell_id

    @ti.kernel
    def cpm(self):
        for i, j in self.grid:
            if self.grid[i, j].is_membrane and ti.random() < self.select_prob:
                # Pick a neighbor
                neigh = self.pick_random_neighbor(i, j)
                # Random swap direction
                s_i, s_j = i, j
                t_i, t_j = neigh[0], neigh[1]
                if ti.random() < .5:
                    s_i, s_j = neigh[0], neigh[1]
                    t_i, t_j = i, j                  
                
                # Mark as selected
                self.grid[s_i, s_j].selected_as_src = 1
                self.grid[s_i, s_j].copy_into = ti.Vector([t_i, t_j], int)
                self.grid[t_i, t_j].selected_as_trg = 1
                self.grid[t_i, t_j].copy_from = ti.Vector([s_i, s_j], int)
            


    def cpm_step(self):
        self.reset_grid_temps()
        self.find_membranes()
        self.cpm()
        self.do_copy()

    def update(self):
        self.cpm_step()
    
    @ti.kernel
    def render(self):
        for i, j in self.grid:
            self.render_grid[i, j, 0] = self.grid[i, j].cell_id
            self.render_grid[i, j, 1] = self.grid[i, j].copy_into[0] / self.size
            self.render_grid[i, j, 2] = self.grid[i, j].copy_into[1] / self.size

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

sim = Simulation()
sim.run()