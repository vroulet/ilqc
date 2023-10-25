import torch
import sys

torch.set_default_tensor_type(torch.DoubleTensor)

sys.path.append('..')
sys.path.append('.')

from envs.pendulum import Pendulum, CartPendulum
from envs.car import Car
from algorithms.run_min_algo import run_min_algo
from finite_horizon_control.fhc_pipeline import solve_ctrl_pb_incrementally, make_env


def recorded_example(exp='pendulum'):
    if exp == 'pendulum':
        env_cfg = dict(env='pendulum', horizon=40, stay_put_time=1.)
    elif exp == 'cart_pendulum':
        env_cfg = dict(env='cart_pendulum', horizon=50, stay_put_time=0.6, x_limits=(-2., 2.), dt=0.05)
    elif exp == 'simple_car':
        env_cfg = dict(env='simple_car', discretization='rk4_cst_ctrl', cost='exact', reg_bar=0.,
                       track='simple', horizon=50)
    elif exp == 'real_car':
        env_cfg = dict(env='real_car', track='simple', discretization='rk4_cst_ctrl', cost='contouring', horizon=50)
    else:
        raise NotImplementedError
    optim_cfg = dict(algo='ddp_linquad_reg', max_iter=50)
    exp_outputs = solve_ctrl_pb_incrementally(env_cfg, optim_cfg)
    cmd_opt = exp_outputs['cmd_opt']
    env = make_env(env_cfg)
    env.visualize(cmd_opt)


def plain_example(exp='pendulum'):
    if exp == 'pendulum':
        env = Pendulum(horizon=40, stay_put_time=1.)
    elif exp == 'cart_pendulum':
        env = CartPendulum(horizon=50, stay_put_time=0.6, x_limits=(-2., 2.), dt=0.05)
    elif exp == 'simple_car':
        env = Car(model='simple', discretization='rk4_cst_ctrl', cost='exact', reg_bar=0.,
                  track='simple', horizon=50)
    elif exp == 'real_car':
        env = Car(model='real', track='simple', discretization='rk4_cst_ctrl', cost='contouring', horizon=50)
    else:
        raise NotImplementedError
    cmd_opt, _, _ = run_min_algo(env, algo='ddp_linquad_reg', max_iter=20)
    env.visualize(cmd_opt)


if __name__ == '__main__':
    # for exp in ['pendulum', 'cart_pendulum', 'simple_car', 'real_car']:
    #     recorded_example(exp)

    for exp in ['pendulum', 'cart_pendulum', 'simple_car', 'real_car']:
        plain_example(exp)


