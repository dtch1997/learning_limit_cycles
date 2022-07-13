import numpy as np
from numpy.random import default_rng
from torch_vector_field import vector_field, utilities, data_utils

# Choose mu relatively high to emphasize non-convexity of loop.
mu = 3.0

def van_der_pol(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[...,0], x[...,1]
    dx1 = x2
    dx2 = mu * (1 - x1 ** 2) * x2 - x1
    dx = np.stack([dx1, dx2], axis=-1)
    return dx

dataset_name = 'van_der_pol'
v = vector_field.FunctionalVectorField(van_der_pol)

# Data generation follows settings in Appendix E3 of https://arxiv.org/pdf/2006.08935.pdf

def generate_train_data():
    """Generate paired (x, dot x) on 20x20 grid"""
    xs = utilities.get_grid_coordinates((-2.5, 2.5), (-4.5, 4.5), 20, 20)
    dot_xs = v.get_gradient(xs)
    return xs, dot_xs

def generate_val_data():
    """Generate paired (x, dot x) on 15x15 grid"""
    xs = utilities.get_grid_coordinates((-2, 2), (-4, 4), 15, 15)
    dot_xs = v.get_gradient(xs)
    return xs, dot_xs

def generate_test_data():
    """Generate 20 trajectories of length 400 with step size 0.05"""
    # Make data generation deterministic
    rng = default_rng(0)
    xs = rng.uniform([-3, -5], [3, 5], size=(20,2))
    times = np.arange(400) * 0.05
    x_trajs = []
    for x in xs:
        x_traj = utilities.solve_trajectory_odeint(v, (0, 400 * 0.05), x, t_eval = times)
        x_trajs.append(x_traj)
    xs = np.concatenate(x_trajs)    
    dot_xs = v.get_gradient(dot_xs)

if __name__ == "__main__":

    x_train, dx_train = generate_train_data()
    data_utils.save_dataset(x_train, dx_train, dataset_name + '_train')

    x_val, dx_val = generate_train_data()
    data_utils.save_dataset(x_val, dx_val, dataset_name + '_val')

    x_test, dx_test = generate_train_data()
    data_utils.save_dataset(x_test, dx_test, dataset_name + '_test')
