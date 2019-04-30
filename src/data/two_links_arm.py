from copy import deepcopy

import torch
from torch.nn.parameter import Parameter

# Physical quantities
k1 = 0.035  # moment of inertia of first link
k2 = 0.045  # moment of inertia of second link
m2 = 1  # mass of second link
l1 = 0.3  # length of first link
l2 = 0.33  # length of second link
d2 = 0.16  # distance from joint center to center of mass of the second link

a1 = k1 + k2 + m2 * l1 ** 2
a2 = m2 * l1 * d2
a3 = k2
B = torch.tensor([[0.05, 0.025],
                  [0.025, 0.05]], dtype=torch.double)


class TwolinksArmDynamic(torch.nn.Module):
    """
    Simulated movement of a two-links arm robot, concatenate discretized movement of a two-links arm robot
     controlled by some ctrl variables stored as torch.nn.Parameter
    """

    def __init__(self, initial_state, time_simulation, horizon):
        super(TwolinksArmDynamic, self).__init__()
        self.initial_state = initial_state
        self.dim_ctrl = 2
        self.dim_state = 4
        self.horizon = horizon
        time_step = time_simulation/horizon

        self.layers = torch.nn.Sequential(
            *[TwoLinksArmLayer(time_step, self.dim_ctrl) for _ in range(self.horizon)])

    def forward(self):
        x = self.initial_state
        return self.layers(x)


class TwoLinksArmLayer(torch.nn.Module):
    """
    Discretized dynamic of a two-links arm robot
     see Li and Todorov Iterative linear quadratic regulator design for nonlinear biological movement systems, 2004
    """

    def __init__(self, time_step, dim_control):
        super(TwoLinksArmLayer, self).__init__()
        self.time_step = time_step
        self.control = Parameter(torch.rand(dim_control))

    def forward(self, input):
        cosx2 = torch.cos(input[1])
        sinx2 = torch.sin(input[1])
        Minv_x2_11 = torch.tensor(a3)
        Minv_x2_12 = -(a3 + a2 * cosx2)
        Minv_x2_22 = (a1 + 2 * a2 * cosx2)
        Minv_x2_raw = torch.stack((torch.stack((Minv_x2_11, Minv_x2_12)),
                                   torch.stack((Minv_x2_12, Minv_x2_22))))
        Minv_x2 = Minv_x2_raw/(a3 * (a1 - a3) - a2 ** 2 * cosx2 ** 2)
        C_x3_x4 = torch.stack(
            (-input[3] * (2 * input[2] + input[3]), input[2] ** 2))

        x1x2 = input[0:2] + self.time_step * input[2:4]
        aux = self.control - a2 * sinx2 * C_x3_x4 - torch.mv(B, input[2:4])
        x3x4 = input[2:4] + self.time_step * torch.mv(Minv_x2, aux)
        next_state = torch.cat((x1x2, x3x4))
        return next_state


class TwoLinksArmTargetCost(torch.nn.Module):
    """
    Final cost for a two-links arm setting: aim at reaching a random target generated previously by the dynamics
    """

    def __init__(self, target, reg_speed):
        super(TwoLinksArmTargetCost, self).__init__()
        self.reg_speed = reg_speed
        self.target = polar_to_cartesian(target.data)

    def forward(self, input):
        input = polar_to_cartesian(input)
        target_cost = 0.5 * (input[0:2] - self.target[0:2]).dot(input[0:2] - self.target[0:2]) \
            + 0.5 * self.reg_speed*input[2:4].dot(input[2:4])
        return target_cost


def polar_to_cartesian(x):
    out_pos1 = l1 * torch.cos(x[0]) + l2 * torch.cos(x[0] + x[1])
    out_pos2 = l1 * torch.sin(x[0]) + l2 * torch.sin(x[0] + x[1])
    out_speed1 = - l1 * \
        x[2] * torch.sin(x[0]) - l2 * (x[2] + x[3]) * torch.sin(x[0] + x[1])
    out_speed2 = l1 * x[2] * torch.cos(x[0]) + \
        l2 * (x[2] + x[3]) * torch.cos(x[0] + x[1])
    out = torch.stack((out_pos1, out_pos2, out_speed1, out_speed2))
    return out
