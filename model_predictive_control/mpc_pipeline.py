from copy import deepcopy
import os
import torch

from algorithms.run_min_algo import run_min_algo, check_cvg_status
from envs.choose_env import make_env
from utils_pipeline.record_exp import run_and_record_exp_wrapper, load_exp


results_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_folder = os.path.join(results_folder, 'results')


def mpc_step(env, full_horizon, overlap, max_iter=10, algo='ddp_linquad_reg', optim_on_full_window=False,
             keep_applying_last_ctrl=True, prev_cmd=None):
    curr_horizon = len(prev_cmd) if prev_cmd is not None else 0
    additional_time_steps = full_horizon - curr_horizon

    if prev_cmd is not None:
        if keep_applying_last_ctrl:
            additional_cmd = prev_cmd[-1].repeat(additional_time_steps, 1)
        else:
            additional_cmd = torch.zeros(additional_time_steps, env.dim_ctrl)
        if optim_on_full_window:
            init_cmd = torch.cat((prev_cmd, additional_cmd))
        else:
            init_cmd = torch.cat((prev_cmd[curr_horizon - overlap:], additional_cmd))
            traj, _ = env.roll_out_cmd(prev_cmd)
            env.init_state = traj[curr_horizon - overlap].data
            env.init_time_iter = curr_horizon - overlap

        cmd, _, metrics = run_min_algo(env, max_iter, algo, prev_cmd=init_cmd)
    else:
        cmd, _, metrics = run_min_algo(env, max_iter, algo)

    if prev_cmd is None or optim_on_full_window:
        cmd_opt = cmd.data
    else:
        cmd_opt = torch.cat((prev_cmd[:-overlap], cmd.data))

    return cmd_opt, metrics


def run_mpc_step(env_cfg, optim_cfg, prev_cmd=None):
    env = make_env(env_cfg)
    cmd_opt, metrics = mpc_step(env, prev_cmd=prev_cmd, **optim_cfg)
    exp_outputs = dict(cmd_opt=cmd_opt, metrics=metrics)
    return exp_outputs


output_to_input = dict(cmd_opt='prev_cmd')


def check_exp_fail(exp_outputs):
    return check_cvg_status(exp_outputs['metrics'], verbose=False) == 'diverged'


run_and_record_mpc_step = run_and_record_exp_wrapper(run_mpc_step, output_to_input, check_exp_fail,
                                                     'full_horizon', results_folder)


def run_mpc(env_cfg, optim_cfg):
    print('env_cfg: {0} \noptim_cfg: {1}'.format(env_cfg, optim_cfg))
    exp_outputs = load_exp(dict(env_cfg=env_cfg, optim_cfg=optim_cfg), results_folder)
    exp_done = exp_outputs is not None
    if not exp_done:
        horizon = 0
        max_horizon, window_size, overlap = optim_cfg['full_horizon'], optim_cfg['window_size'], optim_cfg['overlap']
        sliding_time = window_size - overlap
        while horizon < max_horizon:
            temp_optim_cfg = deepcopy(optim_cfg)
            horizon = min(max(horizon + sliding_time, window_size), max_horizon)
            del temp_optim_cfg['window_size']
            temp_optim_cfg['full_horizon'] = horizon
            exp_outputs = run_and_record_mpc_step(env_cfg, temp_optim_cfg)
            if check_exp_fail(exp_outputs):
                print('Maximum horizon: {}'.format(horizon))
                break
    cmd_opt = exp_outputs['cmd_opt']
    return cmd_opt

