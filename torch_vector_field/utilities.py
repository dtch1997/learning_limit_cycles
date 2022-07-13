import numpy as np
import typing

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_vector_field.vector_field import VectorField, PotentialField
from scipy.integrate import solve_ivp

def simulate_trajectory(
    v: VectorField,
    x0: np.ndarray,  
    step_size = 0.01, 
    num_iters = 1000,
    grad_clip = None,
    ascending=False
): 
    """Simulate the trajectory obtained by gradient descent on a surface"""
    grad = None
    x = x0
    sign = 1 if ascending else -1
    xhist = np.zeros((num_iters, x0.shape[0]))
    for i in range(num_iters):
        xhist[i] = x
        grad = v.get_gradient(x)
        if grad_clip:
            x = x + sign * np.clip(step_size * grad, -grad_clip, grad_clip)
        else:
            x = x + sign * step_size * grad
    return xhist

def solve_trajectory_odeint(
    v: VectorField,
    t_span,
    x0: np.ndarray, 
    t_eval,
    **kwargs
):
    f = lambda t, y: v.get_gradient(y)
    soln = solve_ivp(f, t_span, y0 = x0, t_eval = t_eval, **kwargs)
    return soln.y.T

def simulate_trajectories(
    v: VectorField,
    x0s: typing.List[np.ndarray],  
    step_size = 0.01, 
    num_iters = 1000,
    grad_clip = None,
    ascending=False
): 
    """Simulate the trajectory obtained by gradient descent on a surface"""
    xhists = []
    for x0 in x0s:
        h = simulate_trajectory(v, x0, step_size, num_iters, grad_clip, ascending)
        xhists.append(h)
    return xhists

def get_rotational_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]
    ])

def get_grid_coordinates(x_limits, y_limits, x_num_ticks, y_num_ticks):
    x_coords = np.linspace(x_limits[0], x_limits[1], x_num_ticks)
    y_coords = np.linspace(y_limits[0], y_limits[1], y_num_ticks)
    xs, ys = np.meshgrid(x_coords, y_coords)
    coords = np.vstack([xs.ravel(), ys.ravel()]).T
    return coords