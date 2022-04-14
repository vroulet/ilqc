from typing import List
import torch


def lin_quad_backward(traj: List[torch.Tensor], costs: List[torch.Tensor],
                      reg_ctrl: float = 0., reg_state: float = 0.,
                      handle_bad_dir: str = 'flag'):
    horizon = len(traj)-1
    J, j, jcst = costs[-1].hess_state, costs[-1].grad_state, 0.
    J = J + reg_state * torch.eye(J.shape[0])
    feasible = True
    gains = list()
    for t in range(horizon-1, -1, -1):
        A, B = traj[t+1].grad_dyn_state.t(), traj[t+1].grad_dyn_ctrl.t()

        P, p = costs[t].hess_state, costs[t].grad_state
        Q, q = costs[t+1].hess_ctrl, costs[t+1].grad_ctrl
        P = P + reg_state * torch.eye(P.shape[0])
        Q = Q + reg_ctrl * torch.eye(Q.shape[0])

        J, j, jcst, K, k, feasible = bell_step(A, B, J, j, jcst, P, p, Q, q, handle_bad_dir=handle_bad_dir)
        gain = dict(gain_ctrl=K, offset_ctrl=k)
        gains.insert(0, gain)
        if not feasible:
            break
    return gains, jcst, feasible


def quad_backward(traj, costs, reg_ctrl=0., reg_state=0., handle_bad_dir='flag', mode='newton'):
    horizon = len(traj) - 1
    J, j, jcst = costs[-1].hess_state, costs[-1].grad_state, 0.
    J = J + reg_state * torch.eye(J.shape[0])
    lam = j
    feasible = True
    gains = list()
    for t in range(horizon - 1, -1, -1):
        A, B = traj[t + 1].grad_dyn_state.t(), traj[t + 1].grad_dyn_ctrl.t()

        P, p = costs[t].hess_state, costs[t].grad_state
        Q, q = costs[t + 1].hess_ctrl, costs[t + 1].grad_ctrl
        # P = P + torch.matmul(traj[t + 1].hess_dyn_state, lam) + reg_state * torch.eye(P.shape[0])
        # Q = Q + torch.matmul(traj[t + 1].hess_dyn_ctrl, lam) + reg_ctrl * torch.eye(Q.shape[0])
        # R = torch.matmul(traj[t + 1].hess_dyn_state_ctrl, lam)
        if t > 0:
            # If t==0, P is useless anyway, and it seems that traj[0] does not keep its requires_grad=True flag
            P = P + traj[t + 1].hess_dyn_state(lam) + reg_state * torch.eye(P.shape[0])
        R = traj[t + 1].hess_dyn_state_ctrl(lam)
        Q = Q + traj[t + 1].hess_dyn_ctrl(lam) + reg_ctrl * torch.eye(Q.shape[0])

        J, j, jcst, K, k, feasible = bell_step(A, B, J, j, jcst, P, p, Q, q, R, handle_bad_dir=handle_bad_dir)
        gain = dict(gain_ctrl=K, offset_ctrl=k)
        gains.insert(0, gain)

        if mode == 'newton':
            lam = p + A.mv(lam)
        elif mode == 'ddp':
            lam = j
        if not feasible:
            break
    return gains, jcst, feasible


def quad_backward_ddp(traj, costs, reg_ctrl=0., reg_state=0., handle_bad_dir='flag'):
    return quad_backward(traj, costs, reg_ctrl, reg_state, handle_bad_dir, mode='ddp')


def quad_backward_newton(traj, costs, reg_ctrl=0., reg_state=0., handle_bad_dir='flag'):
    return quad_backward(traj, costs, reg_ctrl, reg_state, handle_bad_dir, mode='newton')


def bell_step(A, B, J, j, jcst, P, p, Q, q, R=None, handle_bad_dir='flag'):
    H = Q + B.t().mm(J.mm(B))
    next_J, next_j, rest, K, k = bell_core(A, B, J, j, H, P, p, q, R)
    if rest > 0:
        if handle_bad_dir == 'flag':
            next_J = next_j = next_jcst = K = k = None
            feasible = False
        elif handle_bad_dir == 'modify_hess':
            (lam, U) = torch.eig(H, eigenvectors=True)
            lam = torch.abs(lam[:, 0])
            H = torch.mm(U, torch.mm(torch.diag(lam), U.t()))
            next_J, next_j, rest, K, k = bell_core(A, B, J, j, H, P, p, q, R)
            feasible = True
            assert rest < 0
            next_jcst = jcst + rest
        elif handle_bad_dir == 'let_it_be':
            feasible = True
            next_jcst = jcst + rest
        else:
            raise NotImplementedError
    else:
        feasible = True
        next_jcst = jcst + rest
    return next_J, next_j, next_jcst, K, k, feasible


def bell_core(A, B, J, j, H, P, p, q, R=None):
    if R is None:
        R = torch.zeros(p.shape[0], q.shape[0])
    K = - torch.solve(R.t() + B.t().mm(J.mm(A)), H)[0]
    k = - torch.solve((q + B.t().mv(j)).view(-1, 1), H)[0].view(-1)
    next_J = P + A.t().mm(J.mm(A)) + (R + A.t().mm(J.mm(B))).mm(K)
    next_j = p + A.t().mv(j) + A.t().mv(J.mv(B.mv(k))) + R.mv(k)
    rest = 0.5 * (q + B.t().mv(j)).dot(k)
    return next_J, next_j, rest, K, k


