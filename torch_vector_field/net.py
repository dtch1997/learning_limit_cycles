
import torch
import torch.nn as nn
import torch.nn.functional as F

import functorch

def radial_coordinate_transform(x):
    """
    x is a tensor of shape [...,2] describing 2D coordinates
    Return a tensor of shape [...,1] (radius) and [...,2] (unit vector)
    """
    r = torch.linalg.norm(x, dim=-1, keepdim=True)
    x_unit = x / r
    return r, x_unit

def polar_coordinate_transform(x):
    """
    x is a tensor of shape [...,2] describing 2D coordinates
    Return a tensor of shape [...,1] (radius) and [...,1] (angle)
    """
    r = torch.linalg.norm(x, dim=-1, keepdim=True)
    x1_unit, x2_unit = x[...,0].unsqueeze(-1), x[...,1].unsqueeze(-1)
    theta = torch.atan2(x2_unit, x1_unit)
    return r, theta

def truncated_fourier_basis(t):
    """
    """
    sin_t = torch.sin(t)
    sin_2t = torch.sin(2*t)
    sin_4t = torch.sin(4*t)
    sin_8t = torch.sin(8*t)
    cos_t = torch.cos(t)
    cos_2t = torch.cos(2*t)
    cos_4t = torch.cos(4*t)
    cos_8t = torch.cos(8*t)
    return torch.cat([sin_t, sin_2t, sin_4t, sin_8t, cos_t, cos_2t, cos_4t, cos_8t], axis=-1)


class SimplePotentialField(nn.Module):
    def __init__(self):
        super(SimplePotentialField, self).__init__()

    def forward(self, x):
        r, theta = polar_coordinate_transform(x)
        return 1 / r + r

class SimpleRotationalField(nn.Module):
    def __init__(self):
        super(SimpleRotationalField, self).__init__()

    def forward(self, x):
        # Make magnitude nonnegative
        _, x_unit = radial_coordinate_transform(x)
        
        x1_unit, x2_unit = x_unit[...,0].unsqueeze(-1), x_unit[...,1].unsqueeze(-1)
        v_unit = torch.cat([-x2_unit, x1_unit], dim=-1)
        return v_unit

class GradientWrapper(nn.Module):
    """ Wraps a scalar-valued nn.Module to get the gradient instead of the value """
    def __init__(self, net: nn.Module, positive: bool = True):
        super(GradientWrapper, self).__init__()
        self.net = net
        self.sign = 1 if positive else -1
        net_fn, net_params, net_buffer = functorch.make_functional_with_buffers(net)

        def net_forward_fn(x):
            """ x is assumed not to have the batch dimension """ 
            out = net_fn(net_params, net_buffer, x)
            out = out.squeeze()
            return out

        self.grad_net_fn = functorch.grad(net_forward_fn)
        # vmap lets us spread the application of grad_fn to each sample in the batch
        self.batched_grad_net_fn = functorch.vmap(functorch.grad(net_forward_fn))

    def forward(self, x):
        if len(x.shape) == 1:
            return self.grad_net_fn(x) * self.sign
        elif len(x.shape) >= 2:
            return self.batched_grad_net_fn(x) * self.sign
        else:
            raise Exception(f"Invalid tensor of shape {x.shape} passed")

class LinearCombinationWrapper(nn.Module):
    """Linearly combine two nn Modules"""
    def __init__(self, v1, v2, a1 = 0.5, a2 = 0.5):
        super(LinearCombinationWrapper, self).__init__()
        self.v1 = v1
        self.v2 = v2 
        self.a1 = a1
        self.a2 = a2 

    def forward(self, x):
        return self.a1 * self.v1(x) + self.a2 * self.v2(x)