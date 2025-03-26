import taichi as ti
import math
import numpy as np
from typing import List

ti.init(arch=ti.cpu, debug=False)

# TODO: Make also self.grid sparse

@ti.data_oriented
class Simulation():
    def __init__(self, H=512, W=512, quality=.5, headless=False):
        self.max_cells = 100
        self.max_parts_per_cell = 6000

        # Physics properties
        self.viscosity = 0.02  # Damp velocities to simulate viscosity
        self.E = 5e3 # Young's Modulus (Stiffness)
        self.nu = 0.4 # Poisson's ratio
        self.mu_0 = self.E / (2 * (1 + self.nu))  # Lame parameters (stress-strain)
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)) # Lame parameters (stress-strain)
        self.h = 0.2 # Hardening coefficient (0 for softer, 1 for stiffer)
        self.mu = self.mu_0 * self.h
        self.la = self.lambda_0 * self.h
        self.friction = 0.9  # Friction coefficient

        # Physics variables
        self.n_grid = 128 * quality
        self.dx = 1 / self.n_grid
        self.inv_dx = 1 / self.dx
        self.time = ti.field(dtype=float, shape=())
        self.dt = 1e-4 / quality
        self.def_rho = 1
        self.def_particle_vol = (self.dx * 0.5)**2
        self.def_particle_mass = self.def_rho * self.def_particle_vol  # Default particle mass

        # Simulation properties
        self.H = H
        self.W = W
        # Using old GUI to support Vulkan 1.4 on some machines
        # self.window = ti.ui.Window(name="MycroVerse", res=(H, W), show_window=not headless)
        # self.canvas = self.window.get_canvas()
        self.gui = ti.GUI(name="MycroVerse", res=max(H, W))
        

        self.create_particle_dynamic_list()

        # IDs of the cells in the simulation
        self.cells = []

        # MPM Grids (Physics)
        self.grid_v = ti.Vector.field(2, dtype=float, shape=(self.H, self.W))  # Velocities
        self.grid_m = ti.field(dtype=float, shape=(self.H, self.W))  # Mass

        # External forces
        self.mouse_press = ti.Vector.field(2, dtype=float, shape=())
        self.mouse_release = ti.Vector.field(2, dtype=float, shape=())
        self.mouse_radius = ti.field(dtype=float, shape=())
        self.mouse_radius[None] = 0.1     

        # What is displayed on screen
        self.mode = 0  # 0: Particle, 1: M Grid, 2: V Grid
        self.render_grid = ti.field(dtype=float, shape=(H, W))
    

    def create_particle_dynamic_list(self):
         # Create particles as a dynamic struct
        self.T_particle = ti.types.struct(pos=ti.math.vec2, # Current position
                                          vel=ti.math.vec2, # Current velocity
                                          acc=ti.math.vec2, # Current acceleration
                                          rest=ti.math.vec2, # Rest position relative to the nucleus
                                          C=ti.math.mat2,  # Affine velocity field
                                          F=ti.math.mat2,  # Deformation gradient
                                          volume=float, # Volume of the particle
                                          mass=float, # Mass of the particle
                                          cell_id=int, # ID of the cell the particle belongs to
                                          is_nucleus=int, # Is this particle a nucleus?
                                          nucleus_index=int, # Index of the nucleus in the cell
                                          is_membrane=int, # Is this particle a membrane?
                                          is_attached=int, # Is this particle attached to the substrate (inamovible)
                                          attractor_direction=ti.math.vec2, # Attractor used for cell movement
                                          attractor_strength=float, # Strength of the attractor
                                          ) 
        
        self.S_particles = ti.root.dynamic(ti.i, 10, chunk_size=32)
        self.particles = self.T_particle.field()
        self.S_particles.place(self.particles)

    @ti.kernel
    def p2g(self):
        """
            Particle to Grid for the M2M Solver.
            Transfer particle mass and velocity to the grid to compute forces
        """

        # Reset grid values
        for i, j in self.grid_m:
            self.grid_m[i, j] = 0.0
            self.grid_v[i, j] = ti.Vector.zero(float, 2)

        for p in range(self.particles.length()):
            # Clamp particle position to allow out-of-bounds particles
            #self.particles[p].pos = ti.max(0.0, ti.min(self.particles[p].pos, 1.0 - 1e-12))

            # Integer position of the particle in the cell grid
            base = (self.particles[p].pos * self.inv_dx - 0.5).cast(int)
            # Fractional position of the particle in the cell grid
            fx = self.particles[p].pos * self.inv_dx - base.cast(float)
            
            # Quadratic Interpolation Kernel
            w = [0.5 * (1.5 - fx) ** 2, 
                0.75 - (fx - 1.0) ** 2, 
                0.5 * (fx - 0.5) ** 2]

            # Update deformation gradient
            self.particles[p].F = (ti.Matrix.identity(float, 2) + self.dt * self.particles[p].C) @ self.particles[p].F

            # Compute SVD of the deformation gradient
            U, sig, V = ti.svd(self.particles[p].F)
            J = 1.0
            for d in ti.static(range(2)):
                J *= sig[d, d]
            stress = 2 * self.mu * (self.particles[p].F - U @ V.transpose()) @ self.particles[p].F.transpose() + ti.Matrix.identity(float, 2) * self.la  * J * (J - 1)
            stress = (-self.dt * self.particles[p].volume * 4 * self.inv_dx * self.inv_dx) * stress * 2.
            affine = stress + self.particles[p].mass * self.particles[p].C

            # Transfer mass and velocity to the grid
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i].x * w[j].y
                self.grid_v[base + offset] += weight * (self.particles[p].mass * self.particles[p].vel + affine @ dpos)
                self.grid_m[base + offset] += weight * self.particles[p].mass
                
    @ti.kernel
    def grid_step(self):
        for i,j in self.grid_m:
            if self.grid_m[i, j] > 0:

                self.grid_v[i, j] /= self.grid_m[i, j] # Momentum to velocity
                self.grid_v[i, j] *= ti.exp(-self.viscosity * self.dt / self.grid_m[i,j]) # Apply viscosity at low reynolds number 
                self.grid_v[i, j] -= self.friction * self.grid_v[i, j] * self.dt # Apply friction
                #Gravity
                # Boundary conditions
                if i < 3 and self.grid_v[i, j][0] < 0:
                    self.grid_v[i, j][0] = 0
                if i > self.n_grid - 3 and self.grid_v[i, j][0] > 0:
                    self.grid_v[i, j][0] = 0
                if j < 3 and self.grid_v[i, j][1] < 0:
                    self.grid_v[i, j][1] = 0
                if j > self.n_grid - 3 and self.grid_v[i, j][1] > 0:
                    self.grid_v[i, j][1] = 0

    @ti.kernel
    def g2p(self):
        for p in range(self.particles.length()):
            base = (self.particles[p].pos * self.inv_dx - 0.5).cast(int)
            fx = self.particles[p].pos * self.inv_dx - base.cast(float)
            
            w = [0.5 * (1.5 - fx) ** 2, 
                0.75 - (fx - 1.0) ** 2, 
                0.5 * (fx - 0.5) ** 2]

            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            
            for i, j in ti.static(ti.ndrange(3, 3)):
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j])]
                weight = w[i].x * w[j].y
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.particles[p].vel = new_v
            self.particles[p].C = new_C
            self.particles[p].pos += self.particles[p].vel * self.dt  # Move particle

    @ti.kernel
    def apply_particle_forces(self):
        for p in range(self.particles.length()):
            
            if self.particles[p].is_attached == 0 and self.particles[p].is_membrane == 1:
                # Apply velocity towards the attractor direction
                self.particles[p].vel += self.particles[self.particles[p].nucleus_index].attractor_direction * self.particles[self.particles[p].nucleus_index].attractor_strength

    @ti.kernel
    def apply_random_motion(self):
        for p in range(self.particles.length()):
            
            if self.particles[p].is_attached == 0 and self.particles[p].is_nucleus == 1:
                self.particles[p].attractor_direction = (ti.Vector([ti.random(), ti.random()])*2 - 1).normalized()
                self.particles[p].attractor_strength = 1

    @ti.kernel
    def random_grow(self, cell_id: int, alpha: float):
        for p in range(self.particles.length()):
        
            if self.particles[p].cell_id == cell_id:
                #self.particles[p].F *= ti.exp(-alpha * 100 * self.dt)
                self.particles[p].F *= 1 - alpha

    def update(self):
        self.apply_random_motion()

        # Physics update
        self.apply_particle_forces()
        self.p2g()      # Particle to grid
        self.grid_step()  # Apply grid-based operations (damping, forces)
        self.g2p()      # Grid to particle

    def spawn_cell(self, x: float = .5, y: float = .5, radius:float=.3):
        cell_id = len(self.cells) + 1
        self.add_cell(xc=x, yc=y, radius=radius, cell_id=cell_id)
        self.cells.append(cell_id)
        print(f"Added cell {cell_id}")
    
    @ti.kernel
    def add_cell(self, xc: float, yc: float, radius: float, cell_id: int):
        
        T = 64
        R = 16
        nucleus_index = self.particles.length()
        # Add nucleus
        nucleus = self.T_particle(pos=[xc, yc], 
                                  rest=[0, 0],
                                  volume=self.def_particle_vol,
                                  mass=self.def_particle_mass, 
                                  cell_id=cell_id, 
                                  is_nucleus=1,
                                  nucleus_index=nucleus_index,
                                  is_membrane=0,
                                  C=ti.Matrix.zero(float, 2, 2),
                                  F=ti.Matrix.identity(float, 2), 
                                  )
        
        self.particles.append(nucleus)
      

        for r in range(R):
            inner_radius = radius * (1 - r / R)
            for t in range(T):
                theta = 2 * math.pi * t / T
                x = xc + inner_radius * ti.cos(theta)
                y = yc + inner_radius * ti.sin(theta)
                new_particle = self.T_particle(pos=[x, y], 
                                               rest = ti.Vector([x - xc, y - yc]),
                                               volume=self.def_particle_vol,
                                               mass=self.def_particle_mass, 
                                               cell_id=cell_id, 
                                               is_nucleus=0,
                                               nucleus_index=nucleus_index,
                                               C=ti.Matrix.zero(float, 2, 2),
                                               F=ti.Matrix.identity(float, 2),
                                               is_membrane= int(r==0),
                )
                self.particles.append(new_particle)
        
    @ti.kernel
    def mitosis(self, cell_id: int):
        # Simplified mitosis: Replace a cell with two identical cells of half the size
        # Find the nucleus of the cell
        nucleus_index = 0
        for p in range(self.particles.length()):
            if self.particles[p].cell_id == cell_id and self.particles[p].is_nucleus == 1:
                ti.atomic_add(nucleus_index, 1)
        # Find the particles of the cell

        # TODO: Implement mitosis
        # Dynamic array does not support deletion
        # Either we recreate the particles array or find a way to reuse previous indices

    def reset(self):
        self.cells = []
        self.create_particle_dynamic_list()

        print(f"Simulation Reset")
        self.time[None] += self.dt
        # Create a grid of particles
        for i, j in zip(range(1, 10), range(1, 10)):
            self.spawn_cell(x=i/10, y=j/10, radius=.03)
    
    @ti.kernel
    def render_density(self):
        """
            Render each pixel as the sum of the particles that are in that position
        """
        self.render_grid.fill(0)
        for p in range(self.particles.length()):
            base = ti.cast(self.particles[p].pos * [self.W, self.H], int)
            ti.atomic_add(self.render_grid[base.x, base.y], 1)

    @ti.kernel
    def render_mass(self):
        """
            Render each pixel as the sum of the particles that are in that position
        """
        self.render_grid.fill(0)
        for p in range(self.particles.length()):
            base = ti.cast(self.particles[p].pos * [self.W, self.H], int)
            ti.atomic_add(self.render_grid[base.x, base.y], self.particles[p].mass)

    @ti.kernel 
    def render_velocity(self):
        """
            Render each pixel as the sum of the particles that are in that position
        """
        self.render_grid.fill(0)
        for p in range(self.particles.length()):
            base = ti.cast(self.particles[p].pos * [self.W, self.H], int)
            ti.atomic_add(self.render_grid[base.x, base.y], self.particles[p].vel.norm()*255)

    def draw(self):
        self.gui.clear()
        # Draw particles
        if self.mode == 0:
            self.render_density()
        elif self.mode == 1:
            self.render_mass()
        elif self.mode == 2:
            self.render_velocity()
        self.gui.set_image(self.render_grid)
        self.gui.show()
    

    def parse_input(self):
        events = self.gui.get_events()
        for e in events:
            if e.key == ti.ui.ESCAPE and e.type.name == ti.ui.PRESS:
                self.gui.destroy()
            if e.key == 'r' and e.type.name == ti.ui.PRESS:
                self.reset()
            if e.key == ti.ui.RMB and e.type.name == ti.ui.PRESS:
                mx, my = self.gui.get_cursor_pos()
                self.spawn_cell(x=mx, y=my, radius=.03)
            if e.key == ti.ui.LMB:
                if e.type.name == ti.ui.PRESS:  # Left Mouse Button Click
                    mx, my = self.gui.get_cursor_pos()  # Get mouse position (0,1)
                    self.mouse_press[None] = ti.Vector([mx, my])
                    print(f"Mouse Pressed at {mx}, {my}")
                else:
                    mx, my = self.gui.get_cursor_pos()  # Get mouse position (0,1)
                    print(f"Mouse Released at {mx}, {my}")
                    self.mouse_release[None] = ti.Vector([mx, my])
                    self.drag_cells_mouse()
            if e.key == ti.ui.UP and e.type.name == ti.ui.PRESS:
                self.mode = (self.mode + 1) % 3
                print(f"Mode: {self.mode}")
            if e.key == ti.ui.DOWN and e.type.name == ti.ui.PRESS:
                self.mode = (self.mode - 1) % 3
                print(f"Mode: {self.mode}")
            if e.key == 'g' and e.type.name == ti.ui.PRESS:
                self.random_grow(1, 0.01)
            

    @ti.kernel
    def drag_cells_mouse(self):
        # Compute the overall drag displacement
        center = self.mouse_press[None]
        disp = self.mouse_release[None] - self.mouse_press[None]
        # For each particle, compute its distance from the mouse press
        for p in range(self.particles.length()):
            d = (self.particles[p].pos - center).norm()
            # Only affect particles within the mouse radius
            if d < self.mouse_radius[None]:
                # Use a weight that decreases linearly with distance from the center.
                # This creates a non-uniform displacement.
                weight = 1.0 - d / self.mouse_radius[None]
                displacement = disp * weight * .1

                # Update the particle's position and reset its velocity/acceleration.
                self.particles[p].pos += displacement
                self.particles[p].vel = displacement / self.dt

                # Estimate a deformation gradient increment.
                # If the displacement were uniformly distributed over the mouse radius,
                # a simple (diagonal) approximation is:
                #grad = ti.Matrix([[displacement.x / self.mouse_radius[None], 0.0],
                #                [0.0, displacement.y / self.mouse_radius[None]]])
                ## Update F: we compose the current deformation with the incremental deformation.
                #self.particles[p].F = (ti.Matrix.identity(float, 2) + grad) @ self.particles[p].F
    
    @ti.kernel
    def apply_force(self, p_id:int, mx: float, my: float):
        
        # Apply velocity towards the click direction
        from_xy = ti.Vector([mx, my]) 
        direction = (from_xy - self.particles[p_id].pos).normalized()
        distance = (from_xy - self.particles[p_id].pos).norm()
        force_strength = 1 / (distance + 1e-3)  # Avoid division by zero
        self.particles[p_id].vel += direction * force_strength * 0.005

    def run(self):
        self.reset()
        while self.gui.running:
            self.parse_input()
            self.update()
            self.draw()

sim = Simulation()
sim.run()