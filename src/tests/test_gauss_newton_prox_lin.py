from src.model.utils_model import conj_grad
from src.model.make_model import make_model
from src.data.get_data import get_data
import torch
import math

torch.set_default_tensor_type(torch.DoubleTensor)


dim_state = 4
dim_ctrl = 2
horizon = 1000

max_iter_conj_grad_factor = 5

reg_ctrl = 0.1
robot, target_cost_func, data_info = get_data(ctrl_setting='linear', dim_state=dim_state,
                                              dim_ctrl=dim_ctrl, horizon=horizon, reg_ctrl=reg_ctrl,
                                              target_goal='quad_random_target')
pb_oracles = make_model(robot, target_cost_func, data_info)


# Test gauss newton
print('Test Gauss-Newton step by comparing solution '
      'found by solving big quad problem and solution proposed through the dual')
func = pb_oracles['func']
grad = pb_oracles['grad']
prox_lin = pb_oracles['prox_lin']

cmd0 = torch.rand(dim_ctrl, horizon)

sol_direct_gn = conj_grad(
    cmd0, grad, max_iter=max_iter_conj_grad_factor * dim_ctrl * horizon)

sol_dual_gn = prox_lin(
    cmd0, math.inf, max_iter_conj_grad_factor=max_iter_conj_grad_factor)

print('Difference btw solution found by big quad problem and dual solution')
print('Norm {}'.format(torch.norm(sol_direct_gn - sol_dual_gn)))
print('All clsose ? {}'.format(sol_direct_gn.allclose(sol_dual_gn)))


# Test prox-lin
print('\nTest prox-lin step by comparing solution '
      'found by solving big quad problem and solution proposed through the dual')
current_cmd = torch.rand(dim_ctrl, horizon)
prox_lin_step_size = 10


def grad_prox_lin(cmd):
    return grad(cmd) + 1/prox_lin_step_size*(cmd-current_cmd)


sol_direct_pl = conj_grad(current_cmd, grad_prox_lin,
                          max_iter=max_iter_conj_grad_factor*dim_ctrl * horizon)

sol_dual_pl = prox_lin(current_cmd, prox_lin_step_size,
                       max_iter_conj_grad_factor=max_iter_conj_grad_factor)

print('Difference btw solution found by big quad problem and dual solution')
print('Norm {}'.format(torch.norm(sol_direct_pl-sol_dual_pl)))
print('All close ? {}'.format(sol_direct_pl.allclose(sol_dual_pl)))
