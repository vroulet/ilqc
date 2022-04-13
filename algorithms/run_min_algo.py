import torch
import math
import time

from algorithms.algos_steps import min_step
from utils_pipeline.optim_helpers import print_info_step


def run_min_algo(env, max_iter=10, algo='ddp_linquad_reg', stepsize=None, obj='neutral', line_search=True,
                 handle_bad_dir='flag', prev_cmd=None, optim_aux_vars=None, past_metrics=None):
    cmd = prev_cmd if prev_cmd is not None else torch.zeros(env.horizon, env.dim_ctrl)
    cmd.requires_grad = True

    algo_type, approx, step_mode = check_nomenclature(algo)

    default_stepsize = 100. if 'reg' in step_mode else 1.
    if optim_aux_vars is not None:
        stepsize = optim_aux_vars['stepsize']
    else:
        stepsize = default_stepsize if stepsize is None else stepsize

    if past_metrics is None:
        traj, costs = env.forward(cmd, approx=approx)
        metrics, iteration = collect_info(None, env, cmd, costs, obj, 0, stepsize, algo, 0.), 0
    else:
        traj, costs = env.forward(cmd, approx=approx)
        metrics, iteration = past_metrics, past_metrics['iteration'][-1]
    status = check_cvg_status(metrics)

    while status == 'running' and iteration < max_iter:
        start_time = time.time()
        cmd, traj, costs, stepsize = min_step(env, traj, costs, cmd, stepsize,
                                              line_search, algo, handle_bad_dir)
        iteration += 1
        time_iter = time.time() - start_time
        metrics = collect_info(metrics, env, cmd, costs, obj, iteration, stepsize, algo, time_iter)
        status = check_cvg_status(metrics)

    cmd_opt = cmd.data if cmd is not None else None
    optim_aux_vars = dict(stepsize=stepsize)
    return cmd_opt, optim_aux_vars, metrics


def collect_info(metrics, env, cmd, costs, obj, iteration, stepsize, algo, time_iter):
    cumul_time = 0. if iteration == 0 else metrics['time'][-1]
    cost = sum(costs) if costs is not None else torch.tensor(float('nan'))

    if algo != 'gd':
        norm_grad_obj = torch.sqrt(sum([cost.grad_ctrl.dot(cost.grad_ctrl) + cost.grad_state.dot(cost.grad_state)
                                        for cost in costs[1:]]))
        _, _, step_mode = algo.split('_')
        if step_mode == 'reg' and iteration > 0:
            stepsize = (stepsize * metrics['norm_grad_obj'][-1]).item()
    else:
        _, costs = env.forward(cmd)
        norm_grad_obj = torch.norm(torch.autograd.grad(sum(costs), cmd)[0])

    info_step = dict(iteration=iteration, time=cumul_time + time_iter, cost=cost.item(),
                     norm_grad_obj=norm_grad_obj.item(), stepsize=stepsize, algo=algo)
    format_types = ['int', 'time', 'scientific', 'scientific', 'scientific', 'string']
    print_info_step(info_step, format_types, iteration == 0)

    if metrics is None:
        metrics = {key: [info_step[key]] for key in info_step.keys()}
    else:
        for key in info_step.keys():
            metrics[key].append(info_step[key])
    return metrics


def check_cvg_status(metrics, verbose=True):
    status = check_cost_status(metrics, verbose)
    if status == 'running':
        status = check_linesearch_status(metrics, verbose)
    return status


def check_linesearch_status(metrics, verbose=True):
    status = 'running'
    if metrics['stepsize'][-1] < 1e-20:
        status = 'got stuck'
        if verbose:
            print('{} got stuck in its linesearch'.format(metrics['algo'][0]))
    return status


def check_cost_status(metrics, verbose=True):
    status = 'running'
    if math.isnan(metrics['cost'][-1]) or metrics['cost'][-1] > metrics['cost'][0] + 1e3:
        status = 'diverged'
        if verbose:
            print('{0} diverged with a cost {1}'.format(metrics['algo'][0], metrics['cost'][-1]))
    if len(metrics['cost']) > 1:
        if abs(metrics['cost'][-1] - metrics['cost'][-2])/abs(metrics['cost'][-2]) < 1e-6:
            status = 'converged'
            if verbose:
                print('{0} converged in {1} iterations'.format(metrics['algo'][0], metrics['iteration'][-1]))
    return status


def check_nomenclature(algo):
    if algo == 'gd':
        algo_type = 'classic'
        approx = None
        step_mode = 'dir'
    else:
        algo_type, approx, step_mode = algo.split('_')
        assert algo_type in ['classic', 'ddp']
        assert approx in ['linquad', 'quad']
        assert step_mode in ['dir', 'reg', 'dirvar', 'regvar']
    return algo_type, approx, step_mode

