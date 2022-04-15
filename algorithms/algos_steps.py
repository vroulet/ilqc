import torch
import numpy as np
from typing import List, Callable

from envs.forward import DiffEnv
from envs.rollout import roll_out_lin, roll_out_exact
from envs.backward import lin_quad_backward, quad_backward_newton, quad_backward_ddp


def gd_oracle(costs: List[torch.Tensor], cmd: torch.Tensor) -> Callable:
    """
    Create an oracle that returns the gradient of the objective and the value of the minimum of the
    associated subproblem
    :param costs: costs computed for the given sequence of control
    :param cmd: current sequence of controls
    :return: oracle that, given a sequence of controls output a direction and the value of the minimum of the
             associated subproblem
    """
    val = sum(costs)
    grad = torch.autograd.grad(val, cmd)[0]
    fixed_obj = -0.5*torch.sum(grad*grad)

    def oracle(stepsize):
        return -stepsize*grad, stepsize*fixed_obj
    return oracle


def classic_oracle(traj: List[torch.Tensor], costs: List[torch.Tensor],
                   approx: str = 'linquad', step_mode: str = 'reg_step', handle_bad_dir: str = 'flag') -> Callable:
    """
    Create an oracle that returns a Gauss-Newton or a Newton direction for the objective and the value of the
    minimum of the associated subproblem
    :param traj: trajectory associated to the current sequence of controls
    :param costs: costs computed for the current sequence of controls
    :param approx: type of approximation used to solve the associated subpb, i.e., 'linquad' for linear-quadratic for a
                   Gauss-Newton step or 'quad' for a quadratic approximation for a Newton step
    :param step_mode: move along a given direction ('dir') or use regularized steps ('reg')
    :param handle_bad_dir: how to handle non-strongly convex objectives see envs.backward.bell_step
    :return: oracle that, given a sequence of controls output a direction and the value of the minimum of the
             associated subproblem
    """
    if approx == 'linquad':
        backward = lin_quad_backward
    elif approx == 'quad':
        backward = quad_backward_newton
    else:
        raise NotImplementedError
    if step_mode in ['dir', 'dirvar']:
        gains, opt_step, feasible = backward(traj, costs, 0., handle_bad_dir=handle_bad_dir)
        reg_ctrl = 1e-6
        while not feasible and reg_ctrl < 1e6:
            gains, opt_step, feasible = backward(traj, costs, reg_ctrl, handle_bad_dir=handle_bad_dir)
            reg_ctrl = 10. * reg_ctrl

        if reg_ctrl > 1e-6:
            print(f'needed to regularize by reg_ctrl={reg_ctrl / 10.}')
            if reg_ctrl >= 1e6:
                print('Failed to find a descent direction')
        diff_cmd = roll_out_lin(traj, gains) if feasible else None

        def oracle(stepsize):
            out = (stepsize*diff_cmd, stepsize*opt_step) if feasible else (None, None)
            return out
    elif step_mode in ['reg', 'regvar']:
        def oracle(stepsize):
            reg = 1/stepsize if stepsize <= 1e6 else 0
            gains, opt_step, feasible = backward(traj, costs, reg, handle_bad_dir=handle_bad_dir)
            out = (roll_out_lin(traj, gains), opt_step) if feasible else (None, None)
            return out
    else:
        raise NotImplementedError
    return oracle


def ddp_oracle(env: DiffEnv, traj: List[torch.Tensor], costs: List[torch.Tensor], cmd: torch.Tensor,
               approx: str = 'linquad', step_mode: str = 'reg_step', handle_bad_dir: str = 'flag') -> Callable:
    """
    Create an oracle that returns a direction computed by a differential dynamic programming approach with either o
    linear-quadratic approximation of the costs or a quadratic approximation of the costs
    :param env: nonlinear control problem environment
    :param traj: trajectory associated to the current sequence of controls
    :param costs: costs computed for the current sequence of controls
    :param cmd: current sequence of controls of shape (horizon, dim_ctrl)
    :param approx: type of approximation used to solve the associated subpb, i.e., 'linquad' for linear-quadratic for a
                   Gauss-Newton step or 'quad' for a quadratic approximation for a Newton step
    :param step_mode: move along a given direction ('dir') or use regularized steps ('reg')
    :param handle_bad_dir: how to handle non-strongly convex objectives see envs.backward.bell_step
    :return: oracle that, given a sequence of controls output a direction and the value of the minimum of the
             associated subproblem
    """
    if approx == 'linquad':
        backward = lin_quad_backward
    elif approx == 'quad':
        backward = quad_backward_ddp
    else:
        raise NotImplementedError
    if step_mode in ['dir', 'dirvar']:
        gains, opt_step, feasible = backward(traj, costs, 0., handle_bad_dir=handle_bad_dir)
        reg_ctrl = 1e-6
        while not feasible and reg_ctrl < 1e6:
            gains, opt_step, feasible = backward(traj, costs, reg_ctrl, handle_bad_dir=handle_bad_dir)
            reg_ctrl = 10. * reg_ctrl
        print(f'needed to regularize by reg_ctrl={reg_ctrl / 10.}')
        if reg_ctrl >= 1e6:
            print('Failed to find a descent direction')

        def oracle(stepsize):
            diff_cmd = roll_out_exact(env, traj, gains, cmd, stepsize=stepsize) if feasible else None
            dec_obj = stepsize * opt_step if feasible else None
            return diff_cmd, dec_obj
    elif step_mode in ['reg', 'regvar']:
        def oracle(stepsize):
            reg = 1/stepsize if stepsize <= 1e6 else 0
            gains, opt_step, feasible = backward(traj, costs, reg, handle_bad_dir=handle_bad_dir)
            out = (roll_out_exact(env, traj, gains, cmd, stepsize=1.), opt_step) if feasible else (None, None)
            return out
    elif step_mode == 'regstate':
        def oracle(stepsize):
            gains, opt_step, feasible = backward(traj, costs, reg_state=1/stepsize, handle_bad_dir=handle_bad_dir)
            out = (roll_out_exact(env, traj, gains, cmd, stepsize=1.), opt_step) if feasible else (None, None)
            return out
    else:
        raise NotImplementedError
    return oracle


def min_step(env: DiffEnv, traj: List[torch.Tensor], costs: List[torch.Tensor], cmd: torch.Tensor, stepsize: float,
             line_search: bool, algo: str, handle_bad_dir: str) -> (torch.Tensor, List[torch.Tensor],
                                                                    List[torch.Tensor], float):
    """
    One step of the given algorithm. Namely, given a sequence of controls, its associated trajectory and costs,
    and a stepsize, return the next candidate sequence of controls with its associated trajectory and costs
    and the stepsize used
    :param env: nonlinear control problem
    :param traj: trajectory associated to the current sequence of controls
    :param costs: costs computed for the current sequence of controls
    :param cmd: current sequence of controls of shape (horizon, dim_ctrl)
    :param stepsize: fixed stepsize or current guess (if a linesearch is used)
    :param line_search: whether to use a line search
    :param algo: algorithm used (see algorithms.run_min_algo.check_nomenclature for the nomenclature of algo)
    :param handle_bad_dir: how to handle non-strongly convex objectives see envs.backward.bell_step
    :return:
        - next_cmd -  next candidate sequence of controls
        - traj - associated trajectory of the next candidate sequence of controls
        - costs - associated costs of the next candidate sequence of controls
        - stepsize - stepsize used to compute the next sequence of controls
    """
    if algo == 'gd':
        oracle = gd_oracle(costs, cmd)
        approx = None
        step_mode = 'regvar'
    else:
        algo_type, approx, step_mode = algo.split('_')
        if algo_type == 'classic':
            oracle = classic_oracle(traj, costs, approx, step_mode, handle_bad_dir)
        elif algo_type == 'ddp':
            oracle = ddp_oracle(env, traj, costs, cmd, approx, step_mode, handle_bad_dir)
        else:
            raise NotImplementedError

    def obj(cmd):
        traj, costs = env.forward(cmd)
        return sum(costs)

    if line_search:
        if step_mode == 'dir':
            # Move along a direction as in classical implementation of e.g. Gauss-Newton/Newton/DDP
            stepsize = 1.
            decrease_fac = 0.9
        elif step_mode == 'reg':
            # Compute a regularized step see the companion report in papers/ilqc_algos
            # Here scale the regularization by the current value of the gradient of the costs
            increase_fac = 8
            norm_grad_obj = torch.sqrt(sum([cost.grad_ctrl.dot(cost.grad_ctrl)
                                            + cost.grad_state.dot(cost.grad_state)
                                            for cost in costs[1:]]))
            stepsize = increase_fac*stepsize/norm_grad_obj
            decrease_fac = 0.5
        elif step_mode == 'regvar':
            # Compute a regularized step see the companion report in papers/ilqc_algos
            # Here use some form of backtracking line-search without scaling
            stepsize *= 8
            decrease_fac = 0.5
        elif step_mode == 'dirvar':
            # Move along a direction with some form of backtracking linesearch
            stepsize = min(4*stepsize, 1)
            decrease_fac = 0.9
        else:
            raise NotImplementedError
        diff_cmd, stepsize = bactrack_line_search(obj, cmd, oracle, stepsize, decrease_fac)
        if step_mode == 'reg':
            stepsize = stepsize*norm_grad_obj
    else:
        diff_cmd, _ = oracle(stepsize)
    if diff_cmd is not None:
        next_cmd = cmd + diff_cmd
        traj, costs = env.forward(next_cmd, approx=approx)
    else:
        next_cmd = traj = costs = None
    return next_cmd, traj, costs, stepsize


def bactrack_line_search(func: Callable, var: torch.Tensor, oracle: Callable, stepsize: float, decrease_fac: float)\
                         -> (torch.Tensor, float):
    """
    Backtracking line search given an objective and a given oracle
    :param func: objective to minimize
    :param var: current candidate solution
    :param oracle: oracle on the objective that returns a direction and a target value to reach for the backtracking
                   linesearch see the companion report in papers/ilqc_algos.pdf
    :param stepsize: initial guess for the stepsize
    :param decrease_fac: factor to decrease the stepsize if the linsearch criterion is not met
    :return:
        - diff_var - candidate direction to update the sequence of controls, if no direction was found, ouput None
        - stepsize - stepsize actually used
    """
    dir = float(np.sign(stepsize))

    curr_val = func(var)
    diff_var, dec_obj = oracle(stepsize)

    if diff_var is None:
        decrease = False
        stop = np.abs(stepsize) < 1e-32
    else:
        new_val = func(var + diff_var)
        decrease = dir * (new_val - (curr_val + dec_obj)) < 0
        stop = torch.norm(diff_var) < 1e-32

    while not decrease and not stop:
        stepsize *= decrease_fac
        diff_var, dec_obj = oracle(stepsize)

        if diff_var is None:
            decrease = False
            stop = np.abs(stepsize) < 1e-32
        else:
            new_val = func(var + diff_var)
            decrease = dir * (new_val - (curr_val + dec_obj)) < 0
            stop = torch.norm(diff_var) < 1e-32

    if stop:
        print('line-search failed')
    return diff_var, stepsize
