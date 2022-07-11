

import icnn
import numpy as np
import torch
from torch.autograd.functional import hessian

def test_icnn():
    """ Test if ICNN is input-convex by evaluating Hessian """
    net_arch = [2,8,1]
    net = icnn.InputConvexNeuralNetwork(net_arch)
    net.reset_parameters()

    sample_inputs = [
        torch.normal(
            mean = torch.zeros(2),
            std = torch.ones(2)
        ) for _ in range(100)
    ]

    for s in sample_inputs:
        h = hessian(lambda x: net(x), s)
        h_np = h.detach().cpu().numpy()
        w, _ = np.linalg.eig(h_np)
        assert np.all(w >= -1e-6)

def test_picnn_shape():
    """ Test that PICNN outputs correct shape """
    x_arch = [2,8,2]
    y_arch = [2,8,1]
    net = icnn.PartiallyInputConvexNeuralNetwork(x_arch, y_arch)
    net.reset_parameters()

    x = torch.zeros(2)
    y = torch.zeros(2)
    assert net(x,y).shape == (1,)

def test_picnn():
    """ Test if PICNN is input-convex by evaluating Hessian """
    y_arch = [2,8,1]
    x_arch = [2,8,2]
    net = icnn.PartiallyInputConvexNeuralNetwork(x_arch, y_arch)
    net.reset_parameters()

    sample_xs = [
        torch.normal(
            mean = torch.zeros(2),
            std = torch.ones(2)
        ) for _ in range(100)
    ]
    sample_ys = [
        torch.normal(
            mean = torch.zeros(2),
            std = torch.ones(2)
        ) for _ in range(100)
    ]

    for x, y in zip(sample_xs, sample_ys):
        h = hessian(lambda x, y: net(x, y), (x, y))[1][1]
        h_np = h.detach().cpu().numpy()
        w, _ = np.linalg.eig(h_np)
        assert np.all(w >= -1e-6)

