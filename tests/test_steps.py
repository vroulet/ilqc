import torch
from envs.car import Car
from envs.backward import lin_quad_backward, quad_backward
from envs.rollout import roll_out_lin

torch.set_default_tensor_type(torch.DoubleTensor)

# Define control problem and candidate control variables
env = Car(model='simple', discretization='euler', cost='exact', horizon=50, dt=0.02)
ctrls = torch.randn(env.horizon, env.dim_ctrl, requires_grad=True)

# Gauss-Newton step
traj, costs = env.forward(ctrls, approx='linquad')
policies = lin_quad_backward(traj, costs)[0]
gauss_newton_dir = roll_out_lin(traj, policies)
gauss_newton_step = ctrls + gauss_newton_dir

# Newton step
traj, costs = env.forward(ctrls, approx='quad')
policies = quad_backward(traj, costs, reg_ctrl=10.)[0]
newton_dir = roll_out_lin(traj, policies)
newton_step = ctrls + newton_dir








