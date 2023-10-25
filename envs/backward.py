from typing import List, Dict, Tuple
import torch


def lin_quad_backward(traj: List[torch.Tensor], costs: List[torch.Tensor],
                      reg_ctrl: float = 0., reg_state: float = 0.,
                      handle_bad_dir: str = 'flag') -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, bool]:
    """
    Compute optimal policies associated to the linear quadratic control problem approximating the environment at the
    current iteration using linear quadratic approximations. See the companion report of the toolbox (
    papers/ilqc_algos.pdf) for a detailed mathematical presentation.
    :param traj: states computed along the input sequence of controls
    :param costs: costs computed along the input sequence of controls
    :param reg_ctrl: regularization to add on the control variables
    :param reg_state: regularization to add on the state variables (akin to a proximal point method)
    :param handle_bad_dir: how to handle the problem if the given costs are not convex such that the resulting
                           subproblem cannot be computed simply by dynamic programming, see function bell_step
    :return:
        - gains - affine policies parameterized by a gain matrix and an offset
        - jcst - optimal value of the linear quadratic control problem used
                 to tune the regularization on the control variables
        - feasible - whether the given lienar quadratic subproblem was feasible or not. If not a larger
                    regularization can make the problem feasible
    """
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


def quad_backward(traj: List[torch.Tensor], costs: List[torch.Tensor], reg_ctrl: float = 0., reg_state: float = 0.,
                  handle_bad_dir: str = 'flag', mode: str = 'newton') \
                  -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, bool]:
    """
    Compute optimal policies associated to the linear quadratic control problem approximating the environment at the
    current iteration using quadratic approximations. See the companion report of the toolbox (
    papers/ilqc_algos.pdf) for a detailed mathematical presentation.
    :param traj: states computed along the input sequence of controls
    :param costs: costs computed along the input sequence of controls
    :param reg_ctrl: regularization to add on the control variables
    :param reg_state: regularization to add on the state variables (akin to a proximal point method)
    :param handle_bad_dir: how to handle the problem if the given costs are not convex such that the resulting
                           subproblem cannot be computed simply by dynamic programming, see function bell_step
    :param mode: whether to instantiate the linear quadratic control problem as done by a Newton step or as done in a
                 differential dynamic programming approach
    :return:
        - gains - affine policies parameterized by a gain matrix and an offset
        - jcst - optimal value of the linear quadratic control problem used
                 to tune the regularization on the control variables
        - feasible - whether the given lienar quadratic subproblem was feasible or not. If not a larger
                    regularization can make the problem feasible
    """
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


def bell_step(A: torch.Tensor, B: torch.Tensor, J: torch.Tensor, j: torch.Tensor, jcst: torch.Tensor,
              P: torch.Tensor, p: torch.Tensor, Q: torch.Tensor, q: torch.Tensor, R: torch.Tensor = None,
              handle_bad_dir: str ='flag') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                               bool]:
    r"""
    Compute cost-to-go function associated to the linear quadratic control problem at a given time step. In other
    words, solve the Bellman equation associated to a linear quadratic control problem. Namely, given the cost-to-go
    at time t+1, the linear dynamic and the qudratic cost defined as, respectively,

    :math: `c_{t+1}(x) = \frac{1}{2} x^\top J_{t+1} x + x^\top j_{t+1}  + j_{t+1, cst}`

    :math: `\ell(x) = A x + B u`

    :math: `h(x, u) = \frac{1}{2} x^\top P x + \frac{1}{2} u^\top Q u + x^\top R u + p^\top x + q^\top u`

    compute the cost-to-go function at time t and the associated policy, defined respectively as

    :math: `c_t(x) = \min_u h(x, u) + c_{t+1}(\ell(x, u))`

    :math: `\pi_t(x) = \argmin_u h(x, u) + c_{t+1}(\ell(x, u))`

    which are parameterized as

    :math: `c_t(x) = \frac{1}{2} x^\top J_t x + x^\top j_t  + j_{t, cst}`

    :math: `\pi_t(x) = K x + k`

    Output the above parameterization after checking whether the step was valid or not

    :param A: see above
    :param B: see above
    :param J: :math: `J_{t+1}`
    :param j: :math: `j_{t+1}`
    :param jcst: :math: `j_{t+1, cst}`
    :param P: see above
    :param p: see above
    :param Q: see above
    :param q: see above
    :param R: see above
    :param handle_bad_dir: the subproblem defining :math: `c_t` may not be strongly convex in u which make the solution
    potentially invalid. To check if the given solution is good, we simply check whether the rest of the minimization
    is negative. If it is not, we either flag it or we modify the associated quadratic function or we just let it be.

    :return:
        - next_J - :math: `J_t`
        - next_j - :math: `j_t`
        - next_jcst - :math: `j_{t, cst}`
        - K - :math: `K`
        - k - :math: `k`
        - feasible - whether the subproblem was feasible
    """
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
    """
    Core linear algebra manipulations (akin to compute a Schur's complement)
    See bell_step for more details.
    """
    if R is None:
        R = torch.zeros(p.shape[0], q.shape[0])
    K = - torch.linalg.solve(H, R.t() + B.t().mm(J.mm(A)))
    k = - torch.linalg.solve(H, (q + B.t().mv(j)))
    next_J = P + A.t().mm(J.mm(A)) + (R + A.t().mm(J.mm(B))).mm(K)
    next_j = p + A.t().mv(j) + A.t().mv(J.mv(B.mv(k))) + R.mv(k)
    rest = 0.5 * (q + B.t().mv(j)).dot(k)
    return next_J, next_j, rest, K, k


