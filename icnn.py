import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing

class NonnegativeLinear(nn.Linear):
    """ Linear layer where weights matrix is constrained to be nonnegative """    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Make weight nonnegative
        weight = F.elu(self.weight, alpha=1.0) + 1
        return F.linear(input, weight, self.bias)


class InputConvexNeuralNetwork(nn.Module):
    """Pytorch implementation of fully input-convex neural networks
    
    Based on description in Fig. 1 of https://arxiv.org/pdf/1609.07152.pdf
    """
    def __init__(self, net_arch: typing.List[int], device=None, dtype=None):
        """ 
        net_arch: [input_dim, hidden_dim_0, ..., hidden_dim_n, output_dim]
        """
        super(InputConvexNeuralNetwork, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        input_dim = net_arch[0]
        nonlinear_layers = []
        passthrough_weights = []
        for d_in, d_out in zip(net_arch[:-1], net_arch[1:]):
            nonlinear_layers.append(
                NonnegativeLinear(d_in, d_out)
            )
            pw = nn.Parameter(torch.empty((d_out, input_dim), **factory_kwargs))
            passthrough_weights.append(pw)
        
        self.nonlinear_layers = nn.ModuleList(nonlinear_layers)
        self.passthrough_weights = nn.ParameterList(passthrough_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        for l, w in zip(self.nonlinear_layers, self.passthrough_weights):
            x = l(x) + F.linear(input, w)
            x = F.elu(x)
        return x

    def reset_parameters(self):
        for l in self.nonlinear_layers:
            l.reset_parameters()
        for w in self.passthrough_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

class InputQuasiconvexNeuralNetwork(InputConvexNeuralNetwork):
    """Pytorch implementation of fully input-quasiconvex neural networks
    
    Assumes nonnegative input
    Adapted by modifying the 'recurring block' of ICNN"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For now, this is exactly the same as ICNN
        # TODO: Experiment with various types of quasiconvexity
        input = x
        for l, w in zip(self.nonlinear_layers, self.passthrough_weights):
            # As QC is a more relaxed condition, 
            # there is potential to insert additional QC terms in the below sum
            x = l(x) + F.linear(input, w)
            # In ICNN, we use elu because it's increasing and convex
            # In IQCNN we only need it to be increasing. 
            # Can experiment with e.g sigmoid. 
            # Consider: https://www.cvxpy.org/tutorial/dqcp/index.html#dqcp-atoms
            x = F.elu(x)
        return x

class PartiallyInputConvexNeuralNetwork(nn.Module):
    """Pytorch implementation of partially input-convex neural networks
    
    Based on description in Fig. 2 of https://arxiv.org/pdf/1609.07152.pdf
    """
    def __init__(self, x_arch: typing.List[int], y_arch: typing.List[int],
        device=None, dtype=None):
        """ 
        net_arch: [input_dim, hidden_dim_0, ..., hidden_dim_n, output_dim]
        """
        assert len(x_arch) == len(y_arch)
        super(PartiallyInputConvexNeuralNetwork, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        y_dim = y_arch[0]

        yy_layers = []
        xx_layers = []
        xy_weights = []
        yyin_weights = []

        for d_x_in, d_x_out, d_y_in, d_y_out in zip(x_arch[:-1], x_arch[1:], y_arch[:-1], y_arch[1:]):
            xx_layers.append(
                nn.Linear(d_x_in, d_x_out)
            )
            yy_layers.append(
                NonnegativeLinear(d_y_in, d_y_out)
            )
            yyin_weights.append(
                nn.Parameter(torch.empty((d_y_out, y_dim), **factory_kwargs))
            )
            xy_weights.append(
                nn.Parameter(torch.empty((d_y_out, d_x_in), **factory_kwargs))
            )
        
        self.yy_layers = nn.ModuleList(yy_layers)
        self.xx_layers = nn.ModuleList(xx_layers)
        self.xy_weights = nn.ParameterList(xy_weights)
        self.yyin_weights = nn.ParameterList(yyin_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ 
        Forward pass is convex in y
        """
        y_in = y
        for l_yy, l_xx, w_yyin, w_xy in zip(
            self.yy_layers, 
            self.xx_layers,
            self.yyin_weights,
            self.xy_weights
        ):
            # Note: this is not the original implementation
            # Original implementation had some additional cross terms
            y = l_yy(y) + F.linear(y_in, w_yyin) + F.linear(x, w_xy)
            y = F.elu(y)
            x = l_xx(x)
            x = F.elu(x)
        return y

    def reset_parameters(self):
        for l in self.xx_layers:
            l.reset_parameters()
        for l in self.yy_layers:
            l.reset_parameters()
        for w in self.yyin_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for w in self.xy_weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))