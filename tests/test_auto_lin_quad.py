import sys
import time

sys.path.append('.')

import torch

from envs.rollout import roll_out_lin
from envs.backward import lin_quad_backward
from envs.lin_quad import LinQuadEnv, make_synth_linear_env
from tests.utils import direct_newton

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(1)


def test_lqr(lin_quad_env: LinQuadEnv, reg_ctrl: float = 0.) -> None:
    """
    Test resolution of linear quadratic regularized control problem by dynamic programming against direct resolution
    by making a Newton step on the flattened problem
    :param lin_quad_env: synthetic linear quadratic control problem as defined in LinQuadEnv
    :param reg_ctrl: regularization on the control variables
    """
    dim_ctrl = lin_quad_env.dim_ctrl

    horizon = len(lin_quad_env.lin_dyn_states)

    cmd0 = torch.rand(horizon, dim_ctrl, requires_grad=True)

    traj, costs = lin_quad_env.forward(cmd0, approx='linquad')
    total_cost0 = sum(costs)
    grad = torch.autograd.grad(total_cost0, cmd0, retain_graph=True)[0]

    print('Solve by Newton step...')
    tic = time.time()
    cmd_opt_newton, opt_newton = direct_newton(lin_quad_env, cmd0, reg_ctrl)
    print(f'Time Newton: {time.time() - tic}')
    if cmd_opt_newton is None:
        print('Newton step failed')
    else:
        cmd_opt_newton = cmd0 + cmd_opt_newton
    val_newton = total_cost0 + opt_newton

    print('Solve by dynamic programming...')
    tic = time.time()
    gains, opt_dyn_prog, feasible = lin_quad_backward(traj, costs, reg_ctrl)
    cmd_opt_dyn_prog = roll_out_lin(traj, gains) if feasible else None
    print(f'Time dynamic programming: {time.time() - tic}')
    if cmd_opt_dyn_prog is None:
        print('Dyn prog failed')
    else:
        cmd_opt_dyn_prog = cmd0 + cmd_opt_dyn_prog
    val_dyn_prog = total_cost0 + opt_dyn_prog

    print_test(cmd_opt_dyn_prog, cmd_opt_newton, grad, val_dyn_prog, val_newton)


def print_test(cmd_opt1: torch.Tensor, cmd_opt2: torch.Tensor, grad: torch.Tensor,
               val1: torch.Tensor = None, val2: torch.Tensor = None):
    """
    Check if solutions are close
    """
    error = torch.norm(cmd_opt1 - cmd_opt2) / torch.norm(cmd_opt1)
    # print('Precision machine is at 1e-16, precision of the algorithms depend then on the norm of the gradient')
    print('Norm gradient by auto-diff : {0:.2e}'.format(torch.norm(grad)))
    print('Relative norm difference solutions: {0:.2e} All close ? {1}'.format(error, cmd_opt1.allclose(cmd_opt2)))
    print('Relative difference optimal value: {0:.2e}\n'.format(torch.abs(val1 - val2)/torch.abs(val1)))


def test():
    reg_ctrl = 10
    dim_state, dim_ctrl, horizon = 4, 4, 50

    lin_quad_env = make_synth_linear_env(horizon, dim_state, dim_ctrl)
    print('Linear quadratic control resolution')
    test_lqr(lin_quad_env)
    print('Regularized step for linear quadratic control problem')
    test_lqr(lin_quad_env, reg_ctrl=reg_ctrl)


if __name__ == '__main__':
    test()



