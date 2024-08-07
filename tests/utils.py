from copy import deepcopy
import sys
import time

sys.path.append('.')

import torch

from envs.lin_quad import LinQuadEnv, compute_lin_quad_approx, DiffEnv
from envs.torch_utils import auto_multi_grad

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(1)

def direct_newton(env: DiffEnv, cmd: torch.Tensor, reg_ctrl: float = 0.) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a Newton step directly by plain automatic differentiation.

    :param env: control environment
    :param cmd: initial command from where to do the Newton step
    :param reg_ctrl: regularization on the control variables, i.e., the command
    :return
        - cmd_opt_newton - optimal command computed by a Newton step
        - opt_newton - optimal value computed by a Newton step
    """
    horizon, dim_ctr = cmd.shape

    # Compute step from a given command
    cmd_flat = deepcopy(cmd.data)
    cmd_flat = cmd_flat.view(-1)
    cmd_flat.requires_grad = True
    cmd_aux = cmd_flat.view(horizon, dim_ctr)

    _, costs = env.forward(cmd_aux)
    total_cost = sum(costs)

    # Compute optimal command
    # Get gradient, hessian, and make a newton step to get the solution
    grad = torch.autograd.grad(total_cost, cmd_flat, create_graph=True)[0]
    # add regularization in the hessian
    hess = auto_multi_grad(grad, cmd_flat) + reg_ctrl * torch.eye(dim_ctr * horizon)
    cmd_opt_newton_flat = - torch.linalg.solve(hess, grad.unsqueeze(-1), ).view(-1)
  
    opt_newton = 0.5 * cmd_opt_newton_flat.dot(grad)
    cmd_opt_newton = cmd_opt_newton_flat.view(horizon, dim_ctr)

    return cmd_opt_newton, opt_newton