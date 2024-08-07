"""Overall pipeline to solve a given control problem."""

import os
from typing import Any, Optional

import torch

from envs.choose_env import make_env
from algorithms.run_min_algo import run_min_algo, check_cvg_status
from utils_pipeline.record_exp import run_exp_incrementally_wrapper

results_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_folder = os.path.join(results_folder, "results")


def solve_ctrl_pb(
    env_cfg: dict[str, Any],
    optim_cfg: dict[str, Any],
    prev_cmd: Optional[torch.Tensor] = None,
    optim_aux_vars: Optional[dict[str, Any]] = None,
    past_metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Overall pipeline to solve a given control problem."""
    env = make_env(env_cfg)
    cmd_opt, optim_aux_vars, metrics = run_min_algo(
        env,
        **optim_cfg,
        prev_cmd=prev_cmd,
        optim_aux_vars=optim_aux_vars,
        past_metrics=past_metrics
    )
    exp_outputs = dict(
        cmd_opt=cmd_opt, optim_aux_vars=optim_aux_vars, metrics=metrics
    )
    return exp_outputs


def check_exp_done(exp_outputs: dict[str, Any]) -> bool:
    return check_cvg_status(exp_outputs["metrics"], verbose=False) != "running"


output_to_input = dict(
    cmd_opt="prev_cmd", optim_aux_vars="optim_aux_vars", metrics="past_metrics"
)

solve_ctrl_pb_incrementally = run_exp_incrementally_wrapper(
    solve_ctrl_pb,
    output_to_input,
    check_exp_done,
    "max_iter",
    10,
    results_folder,
)
