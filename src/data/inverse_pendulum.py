from copy import deepcopy

import torch
from torch.nn.parameter import Parameter

# Physical quantities
g = 9.8  # gravitational force
m = 1  # mass of the pendulum
l = 1  # length of the rope
mu = 0.01  # viscosity constant


class InversePendulumDynamic(torch.nn.Module):
    """
    Simulated movement of a pendulum, concatenate discretized movement of the pendulum controlled by some ctrl
    variables stored as torch.nn.Parameter
    """

    def __init__(self, initial_state, time_simulation, horizon):
        super(InversePendulumDynamic, self).__init__()
        self.initial_state = initial_state
        self.dim_ctrl = 1
        self.dim_state = 2
        self.horizon = horizon
        time_step = time_simulation/horizon

        self.layers = torch.nn.Sequential(*[InversePendulumLayer(time_step, self.dim_ctrl)
                                            for _ in range(self.horizon)])

    def forward(self):
        x = self.initial_state
        return self.layers(x)


class InversePendulumLayer(torch.nn.Module):
    """
    Discretized movement of the pendulum
    """

    def __init__(self, time_step, dim_ctrl):
        super(InversePendulumLayer, self).__init__()
        self.time_step = time_step
        self.control = Parameter(torch.rand(dim_ctrl))

    def forward(self, input):
        sinx0 = torch.sin(input[0])
        next_state = torch.stack((input[0] + self.time_step * input[1],
                                  input[1] + self.time_step *
                                  (g / l * sinx0 - mu / (m * l ** 2) * input[1] + (1 / (m * l ** 2)) *
                                   self.control[0])))
        return next_state


class InversePendulumTargetCost(torch.nn.Module):
    """
    Final cost to enforce the pendulum to swing up
    """

    def __init__(self, reg_speed):
        super(InversePendulumTargetCost, self).__init__()
        self.reg_speed = reg_speed

    def forward(self, input):
        target_cost = 0.5 * \
            (1 + torch.cos(input[0]))**2 + 0.5 * self.reg_speed*input[1]**2
        return target_cost
