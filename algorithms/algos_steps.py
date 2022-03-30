import torch
import numpy as np

from envs.rollout import roll_out_lin, roll_out_exact
from envs.backward import lin_quad_backward, quad_backward_newton, quad_backward_ddp


def gd_oracle(costs, cmd):
    val = sum(costs)
    grad = torch.autograd.grad(val, cmd)[0]
    fixed_obj = -0.5*torch.sum(grad*grad)

    def oracle(stepsize):
        return -stepsize*grad, stepsize*fixed_obj
    return oracle


def classic_oracle(traj, costs, approx='linquad', step_mode='reg_step', handle_bad_dir='flag'):
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


def ddp_oracle(env, traj, costs, cmd, approx, step_mode, handle_bad_dir):
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


def min_step(env, traj, costs, cmd, stepsize, line_search, algo, handle_bad_dir):
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
        traj, costs = env.roll_out_cmd(cmd)
        return sum(costs)

    if line_search:
        if step_mode == 'dir':
            stepsize = 1.
            decrease_fac = 0.9
        elif step_mode == 'reg':
            increase_fac = 8
            norm_grad_obj = torch.sqrt(sum([cost.grad_ctrl.dot(cost.grad_ctrl)
                                            + cost.grad_state.dot(cost.grad_state)
                                            for cost in costs[1:]]))
            stepsize = increase_fac*stepsize/norm_grad_obj
            decrease_fac = 0.5
        elif step_mode == 'regvar':
            stepsize *= 8
            decrease_fac = 0.5
        elif step_mode == 'dirvar':
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
        traj, costs = env.roll_out_cmd(next_cmd, approx=approx)
    else:
        next_cmd = traj = costs = None
    return next_cmd, traj, costs, stepsize


def bactrack_line_search(func, var, oracle, stepsize, decrease_fac):
    dir = float(np.sign(stepsize))

    curr_val = func(var)
    diff_var, dec_obj = oracle(stepsize)

    if diff_var is None:
        decrease = False
        stop = np.abs(stepsize) < 1e-20
    else:
        new_val = func(var + diff_var)
        decrease = dir * (new_val - (curr_val + dec_obj)) < 0
        stop = torch.norm(diff_var) < 1e-12

    while not decrease and not stop:
        stepsize *= decrease_fac
        diff_var, dec_obj = oracle(stepsize)

        if diff_var is None:
            decrease = False
            stop = np.abs(stepsize) < 1e-20
        else:
            new_val = func(var + diff_var)
            decrease = dir * (new_val - (curr_val + dec_obj)) < 0
            stop = torch.norm(diff_var) < 1e-12

    if stop:
        print('line-search failed')
    return diff_var, stepsize
