"""Nonlinear control algorithms runs."""

import math
import time
from typing import Any, Optional

import torch

from envs.forward import DiffEnv
from algorithms.algos_steps import min_step
from utils_pipeline.optim_helpers import print_info_step
from envs.torch_utils import auto_multi_grad


def run_min_algo(
    env: DiffEnv,
    max_iter: int = 10,
    algo: str = "ddp_linquad_reg",
    stepsize: Optional[float] = None,
    line_search: bool = True,
    handle_bad_dir: str = "check_total_value",
    compute_min_sev: bool = False,
    prev_cmd: Optional[torch.Tensor] = None,
    optim_aux_vars: Optional[dict[str, Any]] = None,
    past_metrics: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Runs a given nonlinear control algorithm for a given nonlinear control environment.

    Outputs a candidate optimal, sequence of controls, auxiliary variables to restart the optimization process,
    and some metrics about the convergence process

    Args:
      env: nonlinear control environment
      max_iter: maximum number of iterations for the algorithm
      algo: algorithm given as 'gd' for gradient descent or of the form 'algotype_approx_stepmode'
        where algotype in ['classic, 'ddp'], approx in ['linquad', 'quad'] and stepmode in ['dir', 'reg'],
        see the companion report papers/ilqc_algos.pdf for more details on the algorithms
      stepsize: initial stepsize for the method
      line_search: whether to use a line-search
      handle_bad_dir: how to handle non-strongly convex subproblems see envs.backward.bell_step
      prev_cmd: previously computed candidate sequence of controllers
      optim_aux_vars: previous auxiliary variables of the optimizer when restarting it
      past_metrics: past metrics computed on the problem when restarting the algorithm

    Returns:
      - cmd_opt: final sequence of controllers computed by the algorithm
      - optim_aux_vars: current auxiliary variables of the optimizer such as the stepsize
      - metrics: metrics on the problem
    """
    cmd = (
        prev_cmd
        if prev_cmd is not None
        else torch.zeros(env.horizon, env.dim_ctrl)
    )
    cmd.requires_grad = True

    _, approx, step_mode = check_nomenclature(algo)

    default_stepsize = 100.0 if "reg" in step_mode else 1.0
    if optim_aux_vars is not None:
        stepsize = optim_aux_vars["stepsize"]
    else:
        stepsize = default_stepsize if stepsize is None else stepsize

    if past_metrics is None:
        traj, costs = env.forward(cmd, approx=approx)
        metrics, iteration = (
            collect_info(
                None, env, cmd, costs, 0, stepsize, algo, 0.0, compute_min_sev
            ),
            0,
        )
    else:
        traj, costs = env.forward(cmd, approx=approx)
        metrics, iteration = past_metrics, past_metrics["iteration"][-1]
    status = check_cvg_status(metrics)

    while status == "running" and iteration < max_iter:
        start_time = time.time()
        cmd, traj, costs, stepsize = min_step(
            env, traj, costs, cmd, stepsize, line_search, algo, handle_bad_dir
        )
        iteration += 1
        time_iter = time.time() - start_time
        metrics = collect_info(
            metrics,
            env,
            cmd,
            costs,
            iteration,
            stepsize,
            algo,
            time_iter,
            compute_min_sev,
        )
        status = check_cvg_status(metrics)

    cmd_opt = cmd.data if cmd is not None else None
    optim_aux_vars = dict(stepsize=stepsize)
    return cmd_opt, optim_aux_vars, metrics


def compute_min_sing_eigval_jac(cmd: torch.Tensor, env: DiffEnv) -> float:
    """Computes minimal singular value of the Jacobian of the movement.
    
    Args:
      cmd: sequence of control variables
      env: environment considered

    Returns:
      minimal singular value of the Jacobian of the movement
    """
    cmd_shape = cmd.shape

    flat_cmd = cmd.view(-1)
    unflat_cmd = flat_cmd.view(*cmd_shape)
    traj, _ = env.forward(unflat_cmd)
    traj_ = torch.stack(traj[1:])
    flat_traj = traj_.view(-1)

    jac_t = auto_multi_grad(flat_traj, flat_cmd)

    _, s, _ = torch.svd(jac_t.t().mm(jac_t))
    return min(s).item()


def collect_info(
    metrics: Optional[dict[str, Any]],
    env: DiffEnv,
    cmd: torch.Tensor,
    costs: list[torch.Tensor],
    iteration: int,
    stepsize: float,
    algo: str,
    time_iter: float,
    compute_min_sev: bool,
) -> dict[str, Any]:
    """Collect metric on the problem and add it to the current metrics.

    Args:
      metrics: metrics of the optimization process computed so far
      env: nonlinear control problem environment
      cmd: candidate sequence of controls
      costs: costs associated to the current sequence of controls
      iteration: current iteration
      stepsize: current stepsize
      algo: algorithm used
      time_iter: time of the current iteration

    Returns:
      updated metrics on the optimization process
    """
    cumul_time = 0.0 if iteration == 0 else metrics["time"][-1]
    cost = sum(costs) if costs is not None else torch.tensor(float("nan"))

    _, costs = env.forward(cmd)
    norm_grad_obj = torch.norm(torch.autograd.grad(sum(costs), cmd)[0])
    if not ((algo == "gd") or ("newton" in algo)):
        _, _, step_mode = algo.split("_")
        if step_mode == "reg" and iteration > 0:
            stepsize = (stepsize * metrics["norm_grad_obj"][-1]).item()

    if compute_min_sev:
        min_sev_jac = compute_min_sing_eigval_jac(cmd, env)

    info_step = dict(
        iteration=iteration,
        time=cumul_time + time_iter,
        cost=cost.item(),
        norm_grad_obj=norm_grad_obj.item(),
        stepsize=stepsize,
        algo=algo,
    )
    format_types = [
        "int",
        "time",
        "scientific",
        "scientific",
        "scientific",
        "string",
    ]
    if compute_min_sev:
        info_step.update(min_sev_jac=min_sev_jac)
        format_types.append["scientific"]
    print_info_step(info_step, format_types, iteration == 0)

    if metrics is None:
        metrics = {key: [info_step[key]] for key in info_step.keys()}
    else:
        for key in info_step.keys():
            metrics[key].append(info_step[key])
    return metrics


def check_cvg_status(metrics: dict, verbose: bool = True) -> str:
    """Check if the algorithm converged/diverged/got stuck in its linesearch.

    Args:
      metrics: metrics of the optimization process
      verbose: whether to print if the process converged/diverged/got stuck

    Returns:
      status, i.e., 'running', 'got_stuck', 'converged' or 'diverged'
    """
    status = check_cost_status(metrics, verbose)
    if status == "running":
        status = check_linesearch_status(metrics, verbose)
    return status


def check_linesearch_status(metrics: dict[str, Any], verbose: bool = True) -> str:
    """Check if the linesearch got stuck.

    Args:
      metrics: metrics of the optimization process
      verbose: whether to print if the process converged/diverged/got stuck

    Returns:
      status, i.e., 'running' or 'got_stuck'
    """
    status = "running"
    if metrics["stepsize"][-1] < 1e-24:
        status = "got stuck"
        if verbose:
            print("{} got stuck in its linesearch".format(metrics["algo"][0]))
    return status


def check_cost_status(metrics: dict[str, Any], verbose: bool = True) -> str:
    """Check if the algorithm diverged/converged.

    Args:
      metrics: metrics of the optimization process
      verbose: whether to print if the process converged/diverged/got stuck

    Returns:
      status, i.e., 'running', 'converged' or 'diverged'
    """
    status = "running"
    if (
        math.isnan(metrics["cost"][-1])
        or metrics["cost"][-1] > metrics["cost"][0] + 1e3
    ):
        status = "diverged"
        if verbose:
            print(
                "{0} diverged with a cost {1}".format(
                    metrics["algo"][0], metrics["cost"][-1]
                )
            )
    if len(metrics["cost"]) > 1:
        if (
            abs(metrics["cost"][-1] - metrics["cost"][-2])
            / abs(metrics["cost"][-2])
            < 1e-24
        ):
            status = "converged"
            if verbose:
                print(
                    "{0} converged in {1} iterations".format(
                        metrics["algo"][0], metrics["iteration"][-1]
                    )
                )
    return status


def check_nomenclature(algo: str) -> tuple[str, str, str]:
    """Check if the algorithm given in the input follows the syntax of the toolbox.

    Args:
      algo: given name of the algorithm

    Returns:
      - algo_type: type of algorithm, i.e., 'classic' or 'ddp'
      - approx: type of approximation used to computed the oracles, i.e, 'linquad' or 'quad'
      - step_mode: type of step to use, i.e., move along a given direction or use regualrized steps, i.e.,
        'dir' or 'reg'
    """
    if algo == "gd":
        algo_type = "classic"
        approx = None
        step_mode = "dir"
    elif "newton" in algo:
        algo_type, step_mode = algo.split("_")
        approx = None
        assert step_mode in ["dir", "reg", "dirvar", "regvar"]
    else:
        algo_type, approx, step_mode = algo.split("_")
        assert algo_type in ["classic", "ddp"]
        assert approx in ["linquad", "quad"]
        assert step_mode in ["dir", "reg", "dirvar", "regvar"]
    return algo_type, approx, step_mode
