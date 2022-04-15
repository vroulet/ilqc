from typing import List, Dict
from envs.forward import DiffEnv
import torch


def roll_out_lin(traj: List[torch.Tensor], gains: List[Dict[str, torch.Tensor]], stepsize: float = 1.) -> torch.Tensor:
    """
    Roll-out the given policies given in gains along the linearized trajectories recorded in traj.
    An additional stepsize can be used to perform some line-searches, see the companion report papers/ilqc_algos.pdf
    :param traj: states computed for a given sequence of controls with the linearizations stored as attributes of
                 the states
    :param gains: policies parameterized as :math: `\pi_t(x) = K_t x +k_t`
                  with K_t = gains[t]['gain_ctrl'] and k_t = gains[t]['offset_ctrl'], see envs.backward.bell_step
                  to understand how these policies are computed
    :param stepsize: stepsize for the oracle, see the companion report papers/ilqc_algos.pdf
    :return: oracle direction associated to the policies rolled-out on the linearized dynamics
    """
    dim_state, horizon = traj[0].shape, len(traj)-1
    dim_var = gains[0]['offset_ctrl'].shape[0]

    diff_x = torch.zeros(dim_state)
    diff_vars = torch.zeros(horizon, dim_var)

    for t in range(horizon):
        A = traj[t + 1].grad_dyn_state.t()
        B = traj[t + 1].grad_dyn_ctrl.t()

        diff_var = gains[t]['gain_ctrl'].mv(diff_x) + stepsize*gains[t]['offset_ctrl']
        diff_vars[t] = diff_var
        diff_x = A.mv(diff_x) + B.mv(diff_var)
    return diff_vars


def roll_out_exact(env: DiffEnv, traj: List[torch.Tensor], gains: List[Dict[str, torch.Tensor]], cmd: torch.Tensor,
                   stepsize: float = 1.) -> torch.Tensor:
    """
    Roll-out the given policies given in gains along the original dynamics of teh system.
    An additional stepsize can be used to perform some line-searches, see the companion report papers/ilqc_algos.pdf
    :param env: original system like a pendulum or a car
    :param traj: states computed for a given sequence of controls
    :param gains: policies parameterized as :math: `\pi_t(x) = K_t x +k_t`
    with K_t = gains[t]['gain_ctrl'] and k_t = gains[t]['offset_ctrl']
    :param cmd: sequence of controls used to compute the trajectory
    :param stepsize: stepsize for the oracle, see the companion report papers/ilqc_algos.pdf
    :return: oracle direction associated to the policies rolled-out on the original dynamics
    """
    dim_state, horizon = traj[0].shape, len(traj) - 1
    diff_x, state = torch.zeros(dim_state), env.init_state
    diff_vars = torch.zeros_like(cmd)
    diff_vars.requires_grad = state.requires_grad = False

    for t in range(horizon):
        diff_var = gains[t]['gain_ctrl'].mv(diff_x) + stepsize * gains[t]['offset_ctrl']
        diff_vars[t] = diff_var
        state = env.discrete_dyn(state, cmd[t].data + diff_var, t)
        diff_x = state - traj[t+1].data
    return diff_vars



