import taichi as ti
import math

"""
    Collection of utility functions such as geometric calculations.
    These functions are designed to be used within Taichi kernels and have no side effects.
"""

@ti.func
def local_perimeter(sim, i, j, grid_value) -> int:
    """
        Returns the local perimeter of a pixel in the grid assuming the given grid_value as cell_id.
        (This allows to calculate the perimeter of a pixel assuming it is part of a different cell)

        Args:
            sim: reference to the simulation object
            i, j: coordinates of the pixel
            grid_value: the cell_id to assume for the pixel
        Returns:
            local_perimeter: number of neighboring pixels that are not part of the same cell
    """
    local_perimeter = 0 
    for i_offset in range(-1, 2):
        for j_offset in range(-1, 2):
            if grid_value > 0 and \
            not sim.is_out_of_bounds(i+i_offset, j+j_offset) and \
            grid_value != sim.grid[i+i_offset, j+j_offset].cell_id:
                ti.atomic_add(local_perimeter, 1)
    return local_perimeter

@ti.func
def point_in_polygon(x:int, y:int, polygon: ti.template()) -> int: # type: ignore
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
def get_polygon_edges(xc: float, yc:float, n_edges: int, radius: float, grid_size:int, verts: ti.template()): # type: ignore
    """
    Calculates the vertices of a regular polygon and stores their coordinates in the provided verts array.
    Args:
        xc (float): The x-coordinate of the polygon's center (normalized, between 0 and 1).
        yc (float): The y-coordinate of the polygon's center (normalized, between 0 and 1).
        n_edges (int): The number of edges (vertices) of the polygon.
        radius (float): The radius of the polygon (normalized, between 0 and 1).
        verts (ti.template()): A Taichi field or array to store the computed vertex coordinates. Each entry will be [row, column] in integer pixel coordinates.
    Notes:
        - The coordinates are scaled by grid_size to convert from normalized to pixel space.
        - The vertices are ordered counterclockwise starting from the rightmost point.
    """
    angle_step = 2 * math.pi / n_edges
    for i in range(n_edges):
        angle = i * angle_step
        c = ti.cast((xc + radius * ti.cos(angle))*grid_size, int)
        r = ti.cast((yc + radius * ti.sin(angle))*grid_size, int)
        verts[i] = [r, c]