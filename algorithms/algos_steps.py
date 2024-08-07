"""Algorithm steps."""

from copy import deepcopy
from typing import Any, Callable, Union

import numpy as np
import torch

from envs.torch_utils import auto_multi_grad
from envs.forward import DiffEnv
from envs.rollout import roll_out_lin, roll_out_exact
from envs.backward import (
    lin_quad_backward,
    quad_backward_newton,
    quad_backward_ddp,
)

Scalar = Union[float, torch.Tensor]
Oracle = Callable[[Scalar], tuple[torch.Tensor, Scalar]]


def gd_oracle(costs: list[torch.Tensor], cmd: torch.Tensor) -> Oracle:
    """Gradient oracle.

    Creates an oracle that returns the gradient of the objective and the value of the minimum of the
    associated subproblem

    Args:
      costs: costs computed for the given sequence of control
      cmd: current sequence of controls

    Returns:
      oracle that, given a sequence of controls output a direction and the value of the minimum of the
      associated subproblem
    """
    val = sum(costs)
    grad = torch.autograd.grad(val, cmd)[0]
    fixed_obj = -0.5 * torch.sum(grad * grad)

    def oracle(stepsize):
        return -stepsize * grad, stepsize * fixed_obj

    return oracle


def newton_oracle(
    env: DiffEnv, cmd: torch.Tensor, step_mode: str = "dir"
) -> Oracle:
    """Newton oracle implemented by plain auto-diff (no use of the dynamical structure).
    """
    horizon, dim_ctrl = cmd.shape

    cmd_flat = deepcopy(cmd.data)
    cmd_flat = cmd_flat.view(-1)
    cmd_flat.requires_grad = True
    cmd_aux = cmd_flat.view(horizon, dim_ctrl)

    _, costs = env.forward(cmd_aux)
    total_cost = sum(costs)

    grad_flat = torch.autograd.grad(total_cost, cmd_flat, create_graph=True)[0]
    hess_flat = auto_multi_grad(grad_flat, cmd_flat)

    if "dir" in step_mode:
        reg_ctrl = 0.0
        newton_cmd_flat = - torch.linalg.solve(hess_flat, grad_flat.unsqueeze(-1), ).view(-1)
        newton_opt = 0.5 * newton_cmd_flat.dot(grad_flat)
        feasible = newton_opt < 0.0
        while not feasible:
            reg_ctrl = 10.0 * reg_ctrl if reg_ctrl > 0.0 else 1e-6
            reg_hess_flat = hess_flat + reg_ctrl * torch.eye(
                dim_ctrl * horizon
            )
            newton_cmd_flat = - torch.linalg.solve(reg_hess_flat, grad_flat.unsqueeze(-1), ).view(-1)
            newton_opt = 0.5 * newton_cmd_flat.dot(grad_flat)
            feasible = newton_opt < 0.0
        if reg_ctrl > 1e-6:
            print(f"Newton needed to regularize by reg_ctrl={reg_ctrl}")
            if reg_ctrl > 1e6:
                print("Failed to find a descent direction for newton")
        cmd_newton = newton_cmd_flat.view(horizon, dim_ctrl)

        def oracle(stepsize):
            return stepsize * cmd_newton, stepsize * newton_opt

    elif "reg" in step_mode:

        def oracle(stepsize):
            reg = 1 / stepsize
            reg_hess_flat = hess_flat + reg * torch.eye(dim_ctrl * horizon)
            LD, pivots, _ = torch.linalg.ldl_factor_ex(reg_hess_flat)
            newton_cmd_flat = torch.linalg.ldl_solve(
                LD, pivots, grad_flat.unsqueeze(-1)
            ).view(-1)
            # newton_cmd_flat = - torch.linalg.solve(reg_hess, grad_flat.unsqueeze(-1), ).view(-1)
            newton_opt = 0.5 * newton_cmd_flat.dot(grad_flat)
            return newton_cmd_flat, newton_opt

    else:
        raise NotImplementedError(f"step_mode {step_mode} not implemented yet")

    return oracle


def classic_oracle(
    traj: list[torch.Tensor],
    costs: list[torch.Tensor],
    approx: str = "linquad",
    step_mode: str = "reg",
    handle_bad_dir: str = "flag",
) -> Oracle:
    """Implements classical optimization oracles (Gauss-Newton or Newton) through dynamical structure.

    Creates an oracle that returns a Gauss-Newton or a Newton direction for the objective and the value of the
    minimum of the associated subproblem

    Args:
      traj: trajectory associated to the current sequence of controls
      costs: costs computed for the current sequence of controls
      approx: type of approximation used to solve the associated subpb, i.e., 'linquad' for linear-quadratic for a
        Gauss-Newton step or 'quad' for a quadratic approximation for a Newton step
      step_mode: move along a given direction ('dir') or use regularized steps ('reg')
      handle_bad_dir: how to handle non-strongly convex objectives see envs.backward.bell_step

    Returns:
      oracle that, given a sequence of controls output a direction and the value of the minimum of the
      associated subproblem
    """
    if approx == "linquad":
        backward = lin_quad_backward
    elif approx == "quad":
        backward = quad_backward_newton
    else:
        raise NotImplementedError
    if step_mode in ["dir", "dirvar"]:
        reg_ctrl = 0.0
        gains, opt_step, feasible = backward(
            traj, costs, reg_ctrl, handle_bad_dir=handle_bad_dir
        )
        while not feasible and reg_ctrl < 1e6:
            reg_ctrl = 10.0 * reg_ctrl if reg_ctrl > 0.0 else 1e-6
            gains, opt_step, feasible = backward(
                traj, costs, reg_ctrl, handle_bad_dir=handle_bad_dir
            )

        if reg_ctrl > 1e-6:
            print(
                f"classic {approx} {step_mode} needed to regularize by reg_ctrl={reg_ctrl}"
            )
            if reg_ctrl > 1e6:
                print("Failed to find a descent direction")
        diff_cmd = roll_out_lin(traj, gains) if feasible else None

        def oracle(stepsize):
            out = (
                (stepsize * diff_cmd, stepsize * opt_step)
                if feasible
                else (None, None)
            )
            return out

    elif step_mode in ["reg", "regvar"]:

        def oracle(stepsize):
            reg = 1 / stepsize
            gains, opt_step, feasible = backward(
                traj, costs, reg, handle_bad_dir=handle_bad_dir
            )
            out = (
                (roll_out_lin(traj, gains), opt_step)
                if feasible
                else (None, None)
            )
            return out

    else:
        raise NotImplementedError
    return oracle


def ddp_oracle(
    env: DiffEnv,
    traj: list[torch.Tensor],
    costs: list[torch.Tensor],
    cmd: torch.Tensor,
    approx: str = "linquad",
    step_mode: str = "reg",
    handle_bad_dir: str = "flag",
) -> Oracle:
    """Implements differentiable dynamic programming oracles.

    Create an oracle that returns a direction computed by a differential dynamic programming approach with either o
    linear-quadratic approximation of the costs or a quadratic approximation of the costs

    Args:
      env: nonlinear control problem environment
      traj: trajectory associated to the current sequence of controls
      costs: costs computed for the current sequence of controls
      cmd: current sequence of controls of shape (horizon, dim_ctrl)
      approx: type of approximation used to solve the associated subpb, i.e., 'linquad' for linear-quadratic for a
        Gauss-Newton step or 'quad' for a quadratic approximation for a Newton step
      step_mode: move along a given direction ('dir') or use regularized steps ('reg')
      handle_bad_dir: how to handle non-strongly convex objectives see envs.backward.bell_step

    Returns:
      oracle that, given a sequence of controls output a direction and the value of the minimum of the
      associated subproblem
    """
    if approx == "linquad":
        backward = lin_quad_backward
    elif approx == "quad":
        backward = quad_backward_ddp
    else:
        raise NotImplementedError
    if step_mode in ["dir", "dirvar"]:
        reg_ctrl = 0.0
        gains, opt_step, feasible = backward(
            traj, costs, reg_ctrl, handle_bad_dir=handle_bad_dir
        )
        while not feasible and reg_ctrl < 1e6:
            reg_ctrl = 10.0 * reg_ctrl if reg_ctrl > 0.0 else 1e-6
            gains, opt_step, feasible = backward(
                traj, costs, reg_ctrl, handle_bad_dir=handle_bad_dir
            )
        if reg_ctrl > 1e-6:
            print(
                f"ddp {approx} {step_mode} needed to regularize by reg_ctrl={reg_ctrl}"
            )
            if reg_ctrl > 1e6:
                print("Failed to find a descent direction")

        def oracle(stepsize):
            diff_cmd = (
                roll_out_exact(env, traj, gains, cmd, stepsize=stepsize)
                if feasible
                else None
            )
            dec_obj = stepsize * opt_step if feasible else None
            return diff_cmd, dec_obj

    elif step_mode in ["reg", "regvar"]:

        def oracle(stepsize):
            reg = 1 / stepsize
            # if stepsize <= 1e12 else 0
            gains, opt_step, feasible = backward(
                traj, costs, reg, handle_bad_dir=handle_bad_dir
            )
            out = (
                (roll_out_exact(env, traj, gains, cmd, stepsize=1.0), opt_step)
                if feasible
                else (None, None)
            )
            return out

    elif step_mode == "regstate":

        def oracle(stepsize):
            gains, opt_step, feasible = backward(
                traj,
                costs,
                reg_state=1 / stepsize,
                handle_bad_dir=handle_bad_dir,
            )
            out = (
                (roll_out_exact(env, traj, gains, cmd, stepsize=1.0), opt_step)
                if feasible
                else (None, None)
            )
            return out

    else:
        raise NotImplementedError
    return oracle


def min_step(
    env: DiffEnv,
    traj: list[torch.Tensor],
    costs: list[torch.Tensor],
    cmd: torch.Tensor,
    stepsize: float,
    line_search: bool,
    algo: str,
    handle_bad_dir: str,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], float]:
    """One step of the given algorithm.

    Namely, given a sequence of controls, its associated trajectory and costs,
    and a stepsize, return the next candidate sequence of controls with its associated trajectory and costs
    and the stepsize used
    Args:
      env: nonlinear control problem
      traj: trajectory associated to the current sequence of controls
      costs: costs computed for the current sequence of controls
      cmd: current sequence of controls of shape (horizon, dim_ctrl)
      stepsize: fixed stepsize or current guess (if a linesearch is used)
      line_search: whether to use a line search
      algo: algorithm used (see algorithms.run_min_algo.check_nomenclature for the nomenclature of algo)
      handle_bad_dir: how to handle non-strongly convex objectives see envs.backward.bell_step

    Returns:
      - next_cmd -  next candidate sequence of controls
      - traj - associated trajectory of the next candidate sequence of controls
      - costs - associated costs of the next candidate sequence of controls
      - stepsize - stepsize used to compute the next sequence of controls
    """
    if algo == "gd":
        oracle = gd_oracle(costs, cmd)
        approx = None
        step_mode = "regvar"
    elif "newton" in algo:
        algo_type, step_mode = algo.split("_")
        approx = None
        oracle = newton_oracle(env, cmd)
    else:
        algo_type, approx, step_mode = algo.split("_")
        if algo_type == "classic":
            oracle = classic_oracle(
                traj, costs, approx, step_mode, handle_bad_dir
            )
        elif algo_type == "ddp":
            oracle = ddp_oracle(
                env, traj, costs, cmd, approx, step_mode, handle_bad_dir
            )
        else:
            raise NotImplementedError

    def obj(cmd):
        _, costs = env.forward(cmd)
        return sum(costs)

    if line_search:
        if step_mode == "dir":
            # Move along a direction as in classical implementation of e.g. Gauss-Newton/Newton/DDP
            stepsize = 1.0
            decrease_fac = 0.9
            slope_rtol = 1.0
        elif step_mode == "reg":
            # Compute a regularized step see the companion report in papers/ilqc_algos
            # Here scale the regularization by the current value of the gradient of the costs
            increase_fac = 8
            norm_grad_obj = torch.sqrt(
                sum(
                    [
                        cost.grad_ctrl.dot(cost.grad_ctrl)
                        + cost.grad_state.dot(cost.grad_state)
                        for cost in costs[1:]
                    ]
                )
            )
            stepsize = increase_fac * stepsize / norm_grad_obj
            decrease_fac = 0.5
            slope_rtol = 1.0
        elif step_mode == "regvar":
            # Compute a regularized step see the companion report in papers/ilqc_algos
            # Here use some form of backtracking line-search without scaling
            stepsize *= 8
            decrease_fac = 0.5
            slope_rtol = 1.0
        elif step_mode == "dirvar":
            # Move along a direction with some form of backtracking linesearch
            stepsize = min(4 * stepsize, 1)
            decrease_fac = 0.9
            slope_rtol = 1.0
        else:
            raise NotImplementedError

        diff_cmd, stepsize = backtracking_line_search(
            obj, cmd, oracle, stepsize, decrease_fac, slope_rtol
        )
        if step_mode == "reg":
            stepsize = stepsize * norm_grad_obj
    else:
        diff_cmd, _ = oracle(stepsize)
    if diff_cmd is not None:
        next_cmd = cmd + diff_cmd
        traj, costs = env.forward(next_cmd, approx=approx)
    else:
        next_cmd = traj = costs = None
    return next_cmd, traj, costs, stepsize


def backtracking_line_search(
    fun: Callable[..., Scalar],
    var: torch.Tensor,
    oracle: Oracle,
    stepsize: float,
    decrease_fac: float,
    slope_rtol=1.0,
) -> tuple[torch.Tensor, float]:
    """Backtracking line search given an objective and a given oracle.

    Args:
      fun: objective to minimize
      var: current candidate solution
      oracle: oracle on the objective that returns a direction and a target value to reach for the backtracking
        linesearch see the companion report in papers/ilqc_algos.pdf
      stepsize: initial guess for the stepsize
      decrease_fac: factor to decrease the stepsize if the linsearch criterion is not met

    Returns:
      - diff_var: candidate direction to update the sequence of controls, if no direction was found, ouput None
      - stepsize: stepsize actually used
    """
    dir = float(np.sign(stepsize))
    tol = 0.0

    curr_val = fun(var)
    diff_var, dec_obj = oracle(stepsize)

    if diff_var is None:
        decrease = False
        stop = np.abs(stepsize) < 1e-32
    else:
        new_val = fun(var + diff_var)
        decrease = dir * (new_val - (curr_val + slope_rtol * dec_obj)) < tol
        stop = torch.norm(diff_var) < 1e-32

    while not decrease and not stop:
        stepsize *= decrease_fac
        diff_var, dec_obj = oracle(stepsize)

        if diff_var is None:
            decrease = False
            stop = np.abs(stepsize) < 1e-32
        else:
            new_val = fun(var + diff_var)
            # print(f'decrease: {new_val - (curr_val + slope_rtol*dec_obj)}')
            decrease = (
                dir * (new_val - (curr_val + slope_rtol * dec_obj)) < tol
            )
            stop = torch.norm(diff_var) < 1e-32

    if stop:
        print("line-search failed")
    return diff_var, stepsize
