import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch_vector_field.vector_field import VectorField, PotentialField

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
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.zeros_like(R)
    for iy, ix in np.ndindex(R.shape):
        try:
            value = p.get_value(R[iy, ix])
        except:
            value = p.get_value(np.array([X[iy, ix], Y[iy, ix]]))
        if min_clip or max_clip:
            value = np.clip(value, min_clip, max_clip)
        Z[iy, ix] = value
        
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

def plot_dyn_sys_data(xs, dot_xs):
    """Plot vectors at given positions on vector field"""
    fig, ax = plt.subplots(figsize=(5,5))
    for x, dot_x in zip(xs, dot_xs):
        ax.arrow(
            *x, 
            *(dot_x * 0.2),
            head_width = 0.2,
            head_length = 0.2
        )

def plot_histories(x_hists, **subplot_kwargs):
    fig, ax = plt.subplots(**subplot_kwargs)
    for xhist in x_hists:
        ax.plot(xhist[:,0], xhist[:,1], s=1)
        ax.grid(True)

def plot_section(
    p,
    rs: np.ndarray,
    theta: float = 0,
    **subplot_kwargs
):
    """ Plot a ray along [0, inf] in direction theta """
    fig, ax = plt.subplots(**subplot_kwargs)
    values = []
    for r in rs:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        value = p.get_value(np.array([x,y]))
        values.append(value)
    ax.plot(rs, values)

def plot_sections(
    p,
    rs: np.ndarray,
    thetas: np.ndarray,
    **subplot_kwargs
):
    """ Plot a ray along [0, inf] in direction theta """
    fig, ax = plt.subplots(**subplot_kwargs)
    for theta in thetas:
        values = []
        for r in rs:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            value = p.get_value(np.array([x,y]))
            values.append(value)
        ax.plot(rs, values)