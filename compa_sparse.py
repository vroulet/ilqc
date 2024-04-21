from copy import deepcopy
import sys
import torch
from scipy.sparse.linalg import cg, gmres, LinearOperator
import time
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.float64)

sys.path.append('.')

from envs.rollout import roll_out_lin
from envs.backward import lin_quad_backward
from envs.lin_quad import LinQuadEnv
from envs.torch_utils import auto_multi_grad, auto_multi_hess

def rand_psd(d):
    mat = torch.randn(d, d)
    mat = mat.mm(mat.t())
    return mat


def make_synth_linear_env(horizon=30, dim_state=4, dim_ctrl=3, seed=0):
    torch.random.manual_seed(seed)

    lin_dyn_states = [0.5*torch.randn(dim_state, dim_state) for _ in range(horizon)]
    lin_dyn_ctrls = [0.5*torch.randn(dim_state, dim_ctrl) for _ in range(horizon)]

    quad_cost_states = [torch.zeros(dim_state, dim_state)] + [rand_psd(dim_state) for _ in range(horizon)]
    lin_cost_states = [torch.zeros(dim_state)] + [torch.randn(dim_state) for _ in range(horizon)]
    quad_cost_ctrls = [rand_psd(dim_ctrl) for _ in range(horizon)]
    lin_cost_ctrls = [torch.randn(dim_ctrl) for _ in range(horizon)]

    state0 = torch.randn(dim_state)

    lin_quad_env = LinQuadEnv(lin_dyn_states, lin_dyn_ctrls,
                              quad_cost_states, lin_cost_states,
                              quad_cost_ctrls, lin_cost_ctrls,
                              state0)
    return lin_quad_env


def direct_newton(env, cmd, reg_ctrl, variant='gmres_scipy', time_solve_only=False):
    tic0 = time.time()
    horizon, dim_ctrl = cmd.shape

    # Compute step from a given command
    cmd_flat = deepcopy(cmd.data)
    cmd_flat = cmd_flat.view(-1)
    cmd_flat.requires_grad = True
    cmd_aux = cmd_flat.view(horizon, dim_ctrl)

    _, costs = env.forward(cmd_aux, approx='linquad')
    total_cost = sum(costs)

    # Get gradient, hessian, and make a newton step to get the solution
    grad = torch.autograd.grad(total_cost, cmd_flat, create_graph=True)[0]
    # add regularization in the hessian
    if variant == 'torch_solve':
      hess = auto_multi_grad(grad, cmd_flat) + reg_ctrl * torch.eye(dim_ctrl * horizon)
      tic1 = time.time()
      cmd_opt_newton = - torch.linalg.solve(hess, grad.unsqueeze(-1), ).view(-1)
    elif variant == 'own_cg':
      hvp = make_hvp(grad, cmd_flat)
      hvp_reg = lambda x: hvp(x) + reg_ctrl*x
      tic1 = time.time()
      cmd_opt_newton = own_cg(hvp_reg, grad)
    elif variant == 'gmres_scipy':
      hvp = make_numpy_hvp(grad, cmd_flat)
      hvp_reg = lambda x: hvp(x) + reg_ctrl*x
      hvp_reg = LinearOperator((horizon*dim_ctrl, horizon*dim_ctrl), hvp_reg)
      target = deepcopy(grad.data).numpy()
      tic1 = time.time()
      cmd_opt_newton = cg(hvp_reg, target)[0]
      cmd_opt_newton = torch.from_numpy(cmd_opt_newton)
    else:
        raise NotImplementedError


    cmd_opt_newton = cmd_opt_newton.view(horizon, dim_ctrl)
    time_newton = time.time() - tic1 if time_solve_only else time.time() - tic0
    return cmd_opt_newton, time_newton


def ricatti_based(env, cmd, reg_ctrl, time_solve_only=False):
    tic0 = time.time()
    traj, costs = env.forward(cmd, approx='linquad')
    tic1 = time.time()
    gains, opt_dyn_prog, feasible = lin_quad_backward(traj, costs, reg_ctrl)
    cmd_opt_dyn_prog = roll_out_lin(traj, gains) if feasible else None
    time_dyn_prog = time.time() - tic1 if time_solve_only else time.time() - tic0
    return cmd_opt_dyn_prog, time_dyn_prog


def make_hvp(out, input):
    def hvp(vec):
        return torch.autograd.grad(out, input, vec, retain_graph=True)[0]
    return hvp

def make_numpy_hvp(out, input):
    def hvp(vec):
        vec = torch.from_numpy(vec)
        hvp_vec = torch.autograd.grad(out, input, vec, retain_graph=True)[0]
        return vec.detach().numpy()
    return hvp

def own_cg(matvec, target, init=None, max_iter=100, tol=1e-3):
    v = init if init else torch.zeros_like(target)
    r = target - matvec(v)
    p = r
    r_inner = r.dot(r)
    for t in range(max_iter):
        matvec_p = matvec(p)
        p_matvec_p = p.dot(matvec_p)
        alpha = r_inner/p_matvec_p
        v = v + alpha*p
        r_next = r - alpha*matvec_p
        r_next_inner = r_next.dot(r_next)

        if torch.sqrt(r_next_inner) < tol:
            break
        beta = r_next_inner/r_inner
        p = r_next + beta*p
        r_inner = r_next_inner
        r = r_next
    if torch.sqrt(r_next_inner) > tol:
        print('cg did not converge')
    return v

def test():
    time_solve_only = True
    times_newton = []
    times_dyn_prog = []
    dim_state = 4
    dim_ctrl = 2
    horizons = [2**i for i in range(2, 8)]
    for horizon in horizons:
        env = make_synth_linear_env(horizon, dim_state, dim_ctrl)

        reg_ctrl = 1e-6
        cmd = torch.rand(horizon, dim_ctrl, requires_grad=True)
        cmd_opt_newton, time_newton = direct_newton(env, cmd, reg_ctrl, time_solve_only=time_solve_only)
        print(f'Time Newton: {time_newton}')
        cmd_opt_dyn_prog, time_dyn_prog = ricatti_based(env, cmd, reg_ctrl, time_solve_only=time_solve_only)
        print(f'Time DynProg: {time_dyn_prog}')

        # diff = torch.linalg.norm(cmd_opt_newton-cmd_opt_dyn_prog)
        # print(f'Diff: {diff}')
        times_newton.append(time_newton)
        times_dyn_prog.append(time_dyn_prog) 
    plt.plot(horizons, times_newton)
    plt.plot(horizons, times_dyn_prog)
    plt.legend(['Direct Solve', 'Dynamic Programming'])
    plt.xlabel('Horizon')
    plt.ylabel('Time')
    plt.suptitle('Time comparisons')
    plt.savefig('time_compa_scipy_sparse_solver.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    test()
    


