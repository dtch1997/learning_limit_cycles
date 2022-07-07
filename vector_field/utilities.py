import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vector_field.vector_field import VectorField, PotentialField

def simulate_trajectory(
    v: VectorField,
    x0,  
    step_size = 0.01, 
    num_iters = 1000,
    grad_clip = None
): 
    """Simulate the trajectory obtained by gradient descent on a surface"""
    grad = None
    x = x0
    xhist = np.zeros((num_iters, x0.shape[0]))
    for i in range(num_iters):
        xhist[i] = x
        grad = v.get_gradient(x)
        if grad_clip:
            x = x - jnp.clip(step_size * grad, -grad_clip, grad_clip)
        else:
            x = x - step_size * grad
    return xhist

def plot_potential_field(
    p: PotentialField,
    x_limits,
    y_limits,
    step_size = 0.05,
    min_clip = None,
    max_clip = None
):
    """Plot the 2D surface given by a potential field in R^3 """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(x_limits[0], x_limits[1], step_size)
    Y = np.arange(y_limits[0], y_limits[1], step_size)
    X, Y = jnp.meshgrid(X, Y)
    R = jnp.sqrt(X**2 + Y**2)
    Z = jnp.zeros_like(R)
    for iy, ix in np.ndindex(R.shape):
        value = p.get_value(R[iy, ix])
        if min_clip or max_clip:
            value = jnp.clip(value, min_clip, max_clip)
        Z = Z.at[iy, ix].set(value)
        
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

def plot_vectors(v: VectorField, xs):
    """Plot vectors at given positions on vector field"""
    fig, ax = plt.subplots(figsize=(5,5))
    for x in xs:
        g = v.get_gradient(x)
        ax.arrow(
            *x, 
            *(g * 0.2),
            head_width = 0.2,
            head_length = 0.2
        )

def get_rotational_matrix(theta):
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)], 
        [jnp.sin(theta), jnp.cos(theta)]
    ])