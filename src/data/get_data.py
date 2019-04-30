import torch
import numpy as np
import random

from src.data.two_links_arm import TwolinksArmDynamic, TwoLinksArmTargetCost
from src.data.inverse_pendulum import InversePendulumDynamic, InversePendulumTargetCost
from src.data.linear_dynamics import LinearDynamic
from src.data.utils_data import QuadTargetCost, output_random_target


def get_data(seed=0, horizon=10**3, ctrl_setting='two_links_arm', time_simulation=1, target_goal='quad_random_target',
             dim_state=4, dim_ctrl=2, reg_speed=0.1, reg_ctrl=0.1):
    """
    Simulate a robot from one of the implemented control settings.
    The discretized dynamics are implemented as a torch.nn.Module
     see src.data.two_links_arm, src.data.inverse_pendulum or src.data.linear_dynamics
    The target on the final state is also implemented in Pytorch to backpropagate through
     see src.data.two_links_arm.TwoLinksArmTargetCost, src.data.inverse_pendulum.InversePendulumTargetCost,
         src.data.utils_data.QuadTargetCost
    :return:
        robot: (torch.nn.Module) Simulated discretized movement of robot
        target_cost_func: (torch.nn.Module) Cost on the final state
        data_info: (dict) information on the setting
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if ctrl_setting == 'inverse_pendulum':
        dim_state = 2
        dim_ctrl = 1
        initial_state = torch.ones(dim_state)
        initial_state[1] = 0
        robot = InversePendulumDynamic(initial_state, time_simulation, horizon)
        if target_goal == 'quad_random_target':
            quad_cost = torch.tensor([[1, 0]
                                      [0, reg_speed]], dtype=torch.double)
            random_target = output_random_target(robot)
            target_cost_func = QuadTargetCost(quad_cost, random_target)
        elif target_goal == 'swing_up':
            target_cost_func = InversePendulumTargetCost(reg_speed)
        else:
            raise NotImplementedError

    elif ctrl_setting == 'two_links_arm':
        dim_state = 4
        dim_ctrl = 2
        initial_state = torch.ones(dim_state)
        robot = TwolinksArmDynamic(initial_state, time_simulation, horizon)
        if target_goal == 'quad_random_target':
            quad_cost = torch.tensor([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, reg_speed, 0],
                                      [0, 0, 0, reg_speed]], dtype=torch.double)
            random_target = output_random_target(robot)
            target_cost_func = QuadTargetCost(quad_cost, random_target)
        elif target_goal == 'cartesian_random_target':
            random_target = output_random_target(robot)
            target_cost_func = TwoLinksArmTargetCost(random_target, reg_speed)
        else:
            raise NotImplementedError

    elif ctrl_setting == 'linear':
        initial_state = torch.rand(dim_state)
        robot = LinearDynamic(initial_state, horizon, dim_state, dim_ctrl)

        if target_goal == 'quad_random_target':
            quad_cost = torch.rand(dim_state, dim_state)
            quad_cost = torch.mm(quad_cost, quad_cost.t())
            random_target = output_random_target(robot)
            target_cost_func = QuadTargetCost(quad_cost, random_target)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    data_info = dict(dim_state=dim_state, dim_control=dim_ctrl,
                     horizon=horizon, reg_ctrl=reg_ctrl)

    return robot, target_cost_func, data_info
