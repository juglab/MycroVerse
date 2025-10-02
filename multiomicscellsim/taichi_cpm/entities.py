import taichi as ti
import math

# This defines the maximum number of energy terms we will track per cell, due to a taichi limitation
MAX_ENERGY_TERMS = 10

@ti.dataclass
class CellType():
    j_adhesion_stroma: float
    j_adhesion_other: float
    preferred_volume_stats: ti.math.vec2 # mean and std for preferred volume
    preferred_anisotropy_stats: ti.math.vec2 # mean and std for preferred anisotropy
    preferred_orientation_stats: ti.math.mat2 # mean and std for preferred orientation (2D vector)
    mitosis_age_stats: ti.math.vec2 # mean and std for mitosis age
    
@ti.dataclass
class Cell():
    # Identifiers
    cell_id: int
    cell_type: int
    
    # Preferred parameters
    preferred_volume: float
    preferred_perimeter: float
    # preferred_polarization: ti.math.vec2 # Preferred polarization vector for the cell
    preferred_anisotropy: float # Preferred anisotropy value for the cell (How much the cell is elongated)
    preferred_major_axis: ti.math.vec2 # Preferred major axis (orientation) of the cell's shape

    # Current parameters
    current_volume: float
    current_perimeter: float
    current_anisotropy: float # Current anisotropy value for the cell (How much the cell is elongated)
    current_age: float # Current age of the cell, used for mitosis and other behaviors

    center: ti.math.vec2 # Center of mass
    maj_axis: ti.math.vec2 # Major axis of the cell's shape
    min_axis: ti.math.vec2 # Minor axis of the cell's shape
    max_eigenvalue: float # Maximum eigenvalue of the covariance matrix
    min_eigenvalue: float # Minimum eigenvalue of the covariance matrix
    covariance_matrix: ti.math.mat2 # Covariance matrix for the cell's shape, used for splitting

    # Current energy terms (pre-computed for performance)
    current_energy_terms: ti.types.vector(MAX_ENERGY_TERMS, float) # Array to hold different energy terms. Indices are defined by the order of constraints in the simulation
    current_ellipse_energy: float # Current ellipse energy for the cell
    current_orientation_energy: float # Current orientation energy for the cell
    current_anisotropy_energy: float # Current anisotropy energy for the cell

    # Mitosis Probabilities
    mitosis_prob_volume: float
    mitosis_prob_age: float
    mitosis_age_threshold: float

    # Behavioral parameters
    should_split: int # Flag to indicate if the cell should split (1) or not (0) in the next step
    
    def print_debug(self):
        print(f" Cell {self.cell_id}: Type {self.cell_type}")
        print(f"  Age (current / mitosis thr.): {self.current_age} / {self.mitosis_age_threshold}")
        print(f"  Volume (current / preferred): {self.current_volume} / {self.preferred_volume} (<{self.current_volume / self.preferred_volume if self.preferred_volume > 0 else 0:.2f}>)")
        print(f"  Perimeter (current / preferred): {self.current_perimeter} / {self.preferred_perimeter} (<{self.current_perimeter / self.preferred_perimeter if self.preferred_perimeter > 0 else 0:.2f}>)")
        print(f"  Orientation (current / preferred): {self.maj_axis} / {self.preferred_major_axis} (<{math.degrees(math.atan2(self.maj_axis[1], self.maj_axis[0])):.2f}°>)")
        print(f"  Anisotropy (current / preferred): {self.current_anisotropy} / {self.preferred_anisotropy} (<{self.current_anisotropy / self.preferred_anisotropy if self.preferred_anisotropy > 0 else 0:.2f}>)")
        print(f"  Center: {self.center}")
        print(f"  Major Axis: {self.maj_axis} (Degree: {math.degrees(math.atan2(self.maj_axis[1], self.maj_axis[0]))})")
        print(f"  Minor Axis: {self.min_axis} (Degree: {math.degrees(math.atan2(self.min_axis[1], self.min_axis[0]))})")
        print(f"  Should Split: {self.should_split}")
        print(f"  Covariance Matrix: {self.covariance_matrix}")
        print(f"  Anisotropy: {self.current_anisotropy}")
        print(f"  Current Energy: {self.current_volume_energy + self.current_perimeter_energy + self.current_anisotropy_energy + self.current_orientation_energy}")
        print(f"  Current Volume Energy: {self.current_volume_energy}")
        print(f"  Current Perimeter Energy: {self.current_perimeter_energy}")
        print(f"  Current Anisotropy Energy: {self.current_anisotropy_energy}")
        print(f"  Current Orientation Energy: {self.current_orientation_energy}")
        print(f"  Mitosis Probabilities: Age {self.mitosis_prob_age}, Volume {self.mitosis_prob_volume}")

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
