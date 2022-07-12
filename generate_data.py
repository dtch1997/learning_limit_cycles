import numpy as np
from torch_vector_field import vector_field

def simple_rot_v(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[...,0], x[...,1]
    r = x1 ** 2 + x2 ** 2
    dx1 = x1 - x2 - x1 * r
    dx2 = x1 + x2 - x2 * r
    # Invert direction to accommodate gradient descent
    dx = np.stack([dx1, dx2], axis=-1)
    return dx

step_size = 0.04
grid_lower = -2
grid_upper = 2

if __name__ == "__main__":
    v = vector_field.FunctionalVectorField(simple_rot_v)

    coords = np.arange(grid_lower, grid_upper, step_size)
    n = coords.shape[0]
    print(f"Coordinates: {n} by {n} grid from {grid_lower} to {grid_upper}")
    x1s, x2s = np.meshgrid(coords, coords)
    xs = np.vstack([x1s.ravel(), x2s.ravel()]).T
    dot_xs = v.get_gradient(xs)
    
    with open('data/simple_rot_field_x.npy', 'wb') as file:
        np.save(file, xs)
    with open('data/simple_rot_field_dx.npy', 'wb') as file:
        np.save(file, dot_xs)
