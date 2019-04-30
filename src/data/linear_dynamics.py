import torch
from torch.nn import Parameter


class LinearDynamic(torch.nn.Module):
    """
    Random Linear movement concatenate discretized movement of linear dynamics controlled by some ctrl
    variables stored as torch.nn.Parameter
    """

    def __init__(self, initial_state, horizon, dim_state=4, dim_ctrl=2):
        super(LinearDynamic, self).__init__()
        self.initial_state = initial_state
        self.horizon = horizon
        self.dim_state = dim_state
        self.dim_ctrl = dim_ctrl

        self.layers = torch.nn.Sequential(
            *[LinearLayer(self.dim_state, self.dim_ctrl) for _ in range(horizon)])

    def forward(self):
        x = self.initial_state
        return self.layers(x)


class LinearLayer(torch.nn.Module):
    """
    Random linear discretized dynamic
    """

    def __init__(self, dim_state, dim_ctrl):
        super(LinearLayer, self).__init__()
        self.dim_state = dim_state
        self.dim_ctrl = dim_ctrl
        self.lin_dyn_x = torch.rand(dim_state, dim_state)/10
        self.lin_dyn_u = torch.rand(dim_state, dim_ctrl)/10
        self.ctrl = Parameter(torch.rand(dim_ctrl))

    def forward(self, input):
        return torch.mv(self.lin_dyn_x, input) + torch.mv(self.lin_dyn_u, self.ctrl)
