"""Roll-out control inputs along dynamics or linearized dynamics."""

import torch

from envs.forward import DiffEnv


def roll_out_lin(
    traj: list[torch.Tensor],
    gains: list[dict[str, torch.Tensor]],
    stepsize: float = 1.0,
) -> torch.Tensor:
    """Roll-out the given policies given in gains along the linearized trajectories recorded in traj.

    An additional stepsize can be used to perform some line-searches.
    See [Iterative Linear Quadratic Optimization for Nonlinear Control:
    Differentiable Programming Algorithmic Templates, Roulet et al, 2022](https://arxiv.org/pdf/2207.06362)
    for more details.

    Args:
      traj: states computed for a given sequence of controls with the linearizations stored as attributes of
        the states
      gains: policies parameterized as :math: `\pi_t(x) = K_t x +k_t`
        with K_t = gains[t]['gain_ctrl'] and k_t = gains[t]['offset_ctrl'], see envs.backward.bell_step
        to understand how these policies are computed
      stepsize: stepsize for the oracle, see the companion report papers/ilqc_algos.pdf

    Returns:
      oracle direction associated to the policies rolled-out on the linearized dynamics
    """
    dim_state, horizon = traj[0].shape, len(traj) - 1
    dim_var = gains[0]["offset_ctrl"].shape[0]

    diff_x = torch.zeros(dim_state)
    diff_vars = torch.zeros(horizon, dim_var)

    for t in range(horizon):
        A = traj[t + 1].grad_dyn_state.t()
        B = traj[t + 1].grad_dyn_ctrl.t()

        diff_var = (
            gains[t]["gain_ctrl"].mv(diff_x)
            + stepsize * gains[t]["offset_ctrl"]
        )
        diff_vars[t] = diff_var
        diff_x = A.mv(diff_x) + B.mv(diff_var)
    return diff_vars


def roll_out_exact(
    env: DiffEnv,
    traj: list[torch.Tensor],
    gains: list[dict[str, torch.Tensor]],
    cmd: torch.Tensor,
    stepsize: float = 1.0,
) -> torch.Tensor:
    """Roll-out the given policies given in gains along the original dynamics of the system.

    An additional stepsize can be used to perform some line-searches.
    See [Iterative Linear Quadratic Optimization for Nonlinear Control:
    Differentiable Programming Algorithmic Templates, Roulet et al, 2022](https://arxiv.org/pdf/2207.06362)
    for more details.

    Args:
      env: original system like a pendulum or a car
      traj: states computed for a given sequence of controls
      gains: policies parameterized as :math: `\pi_t(x) = K_t x +k_t`
        with K_t = gains[t]['gain_ctrl'] and k_t = gains[t]['offset_ctrl']
      cmd: sequence of controls used to compute the trajectory
      stepsize: stepsize for the oracle, see the companion report

    Returns:
      oracle direction associated to the policies rolled-out on the original dynamics
    """
    dim_state, horizon = traj[0].shape, len(traj) - 1
    diff_x, state = torch.zeros(dim_state), env.init_state
    diff_vars = torch.zeros_like(cmd)
    diff_vars.requires_grad = state.requires_grad = False

    for t in range(horizon):
        diff_var = (
            gains[t]["gain_ctrl"].mv(diff_x)
            + stepsize * gains[t]["offset_ctrl"]
        )
        diff_vars[t] = diff_var
        state = env.discrete_dyn(state, cmd[t].data + diff_var, t)
        diff_x = state - traj[t + 1].data
    return diff_vars
