import torch
import math
import time
from typing import List

from envs.forward import DiffEnv
from algorithms.algos_steps import min_step
from utils_pipeline.optim_helpers import print_info_step


def run_min_algo(env: DiffEnv, max_iter: int = 10, algo: str = 'ddp_linquad_reg', stepsize: float = None,
                 line_search: bool = True, handle_bad_dir: str = 'flag',
                 prev_cmd: torch.Tensor = None, optim_aux_vars: dict = None,  past_metrics: dict = None)\
                 -> (torch.Tensor, dict, dict):
    """
    Run a given nonlinear control algorithm for a given nonlinear control environment, outputs a candidate optimal
    sequence of controls, auxiliary variables to restart the optimization process and some metrics about the
    convergence process
    :param env: nonlinear control environment
    :param max_iter: maximum number of iterations for the algorithm
    :param algo: algorithm given as 'gd' for gradient descent or of the form 'algotype_approx_stepmode'
                 where algotype in ['classic, 'ddp'], approx in ['linquad', 'quad'] and stepmode in ['dir', 'reg'],
                 see the companion report papers/ilqc_algos.pdf for more details on the algorithms
    :param stepsize: initial stepsize for the method
    :param line_search: whether to use a line-search
    :param handle_bad_dir: how to handle non-strongly convex subproblems see envs.backward.bell_step
    :param prev_cmd: previously computed candidate sequence of controllers
    :param optim_aux_vars: previous auxiliary variables of the optimizer when restarting it
    :param past_metrics: past metrics computed on the problem when restarting the algorithm
    :return:
        - cmd_opt - final sequence of controllers computed by the algorithm
        - optim_aux_vars - current auxiliary variables of the optimizer such as the stepsize
        - metrics - metrics on the problem
    """
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
        metrics, iteration = collect_info(None, env, cmd, costs, 0, stepsize, algo, 0.), 0
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
        metrics = collect_info(metrics, env, cmd, costs, iteration, stepsize, algo, time_iter)
        status = check_cvg_status(metrics)

    cmd_opt = cmd.data if cmd is not None else None
    optim_aux_vars = dict(stepsize=stepsize)
    return cmd_opt, optim_aux_vars, metrics


def collect_info(metrics: dict, env: DiffEnv, cmd: torch.Tensor, costs: List[torch.Tensor], iteration: int,
                 stepsize: float, algo: str, time_iter: float) -> dict:
    """
    Collect metric on the problem and add it to the current metrics
    :param metrics: metrics of the optimization process computed so far
    :param env: nonlinear control problem environment
    :param cmd: candidate sequence of controls
    :param costs: costs associated to the current sequence of controls
    :param iteration: current iteration
    :param stepsize: current stepsize
    :param algo: algorithm used
    :param time_iter: time of the current iteration
    :return: updated metrics on the optimization process
    """
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


def check_cvg_status(metrics: dict, verbose: bool = True) -> str:
    """
    Check if the algorithm converged/diverged/got stuck in its linesearch
    :param metrics: metrics of the optimization process
    :param verbose: whether to print if the process converged/diverged/got stuck
    :return: status, i.e., 'running', 'got_stuck', 'converged' or 'diverged'
    """
    status = check_cost_status(metrics, verbose)
    if status == 'running':
        status = check_linesearch_status(metrics, verbose)
    return status


def check_linesearch_status(metrics: dict, verbose: bool = True) -> str:
    """
    Check if the linesearch got stuck
    :param metrics: metrics of the optimization process
    :param verbose: whether to print if the process converged/diverged/got stuck
    :return: status, i.e., 'running' or 'got_stuck'
    """
    status = 'running'
    if metrics['stepsize'][-1] < 1e-24:
        status = 'got stuck'
        if verbose:
            print('{} got stuck in its linesearch'.format(metrics['algo'][0]))
    return status


def check_cost_status(metrics: dict, verbose: bool = True) -> str:
    """
    Check if the algorithm diverged/converged
    :param metrics: metrics of the optimization process
    :param verbose: whether to print if the process converged/diverged/got stuck
    :return: status, i.e., 'running', 'converged' or 'diverged'
    """
    status = 'running'
    if math.isnan(metrics['cost'][-1]) or metrics['cost'][-1] > metrics['cost'][0] + 1e3:
        status = 'diverged'
        if verbose:
            print('{0} diverged with a cost {1}'.format(metrics['algo'][0], metrics['cost'][-1]))
    if len(metrics['cost']) > 1:
        if abs(metrics['cost'][-1] - metrics['cost'][-2]) / abs(metrics['cost'][-2]) < 1e-12:
        # if abs(metrics['cost'][-1] - metrics['cost'][-2])/abs(metrics['cost'][-2]) < 1e-6:
            status = 'converged'
            if verbose:
                print('{0} converged in {1} iterations'.format(metrics['algo'][0], metrics['iteration'][-1]))
    return status


def check_nomenclature(algo: str) -> (str, str, str):
    """
    Check if the algorithm given in the input follows the syntax of the toolbox
    :param algo: given name of the algorithm
    :return:
        - algo_type -  type of algorithm, i.e., 'classic' or 'ddp'
        - approx - type of approximation used to computed the oracles, i.e, 'linquad' or 'quad'
        - step_mode - type of step to use, i.e., move along a given direction or use regualrized steps, i.e.,
                      'dir' or 'reg'
    """
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

