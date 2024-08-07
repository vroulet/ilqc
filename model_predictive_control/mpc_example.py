import sys

from matplotlib import pyplot as plt
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

sys.path.append("..")
sys.path.append(".")

from model_predictive_control.mpc_pipeline import run_mpc
from envs.choose_env import make_env


def simple_track_example():
    env_cfg = dict(env="real_car", track="simple")
    optim_cfg_mpc = dict(
        algo="ddp_linquad_reg",
        max_iter=10,
        overlap=39,
        window_size=40,
        full_horizon=200,
    )

    cmd_mpc = run_mpc(env_cfg, optim_cfg_mpc)

    env = make_env(env_cfg)
    traj, costs = env.forward(cmd_mpc)
    fig, ax = env.plot_track()
    env.plot_traj(traj, fig, ax)
    plt.show()
    env.visualize(cmd_mpc)


def complex_track_example():
    env_cfg = dict(
        env="real_car",
        track="complex",
        vref=2.5,
        reg_cont=1.0,
        reg_lag=1.0,
        reg_speed=1.0,
    )
    optim_cfg_mpc = dict(
        algo="ddp_linquad_reg",
        max_iter=10,
        overlap=39,
        window_size=40,
        full_horizon=800,
    )

    cmd_mpc = run_mpc(env_cfg, optim_cfg_mpc)

    env = make_env(env_cfg)
    traj, costs = env.forward(cmd_mpc)
    fig, ax = env.plot_track()
    env.plot_traj(traj, fig, ax)
    plt.show()
    env.visualize(cmd_mpc)


if __name__ == "__main__":
    simple_track_example()
    complex_track_example()
