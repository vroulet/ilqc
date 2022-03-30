import torch

from envs.forward import DiffEnv


class LinQuadEnv(DiffEnv):
    def __init__(self, lin_dyn_states, lin_dyn_ctrls,
                 quad_cost_states, lin_cost_states,
                 quad_cost_ctrls, lin_cost_ctrls,
                 init_state=None):
        super(LinQuadEnv, self).__init__()

        # dynamics
        self.lin_dyn_states = lin_dyn_states
        self.lin_dyn_ctrls = lin_dyn_ctrls

        # costs
        self.quad_cost_states = quad_cost_states
        self.lin_cost_states = lin_cost_states
        self.quad_cost_ctrls = quad_cost_ctrls
        self.lin_cost_ctrls = lin_cost_ctrls

        # dimensions:
        self.dim_ctrl = lin_dyn_ctrls[0].shape[1]
        self.dim_state = lin_dyn_states[0].shape[1]
        self.horizon = len(self.lin_dyn_states)

        # initialization
        self.init_time_iter = 0
        self.init_state = init_state if init_state is not None else torch.zeros(self.dim_state)

    def discrete_dyn(self, state, ctrl, time_iter):
        next_state = self.lin_dyn_states[time_iter].mv(state) + self.lin_dyn_ctrls[time_iter].mv(ctrl)
        return next_state

    def costs(self, next_state, ctrl, time_iter):
        cost_next_state = 0.5 * next_state.dot(self.quad_cost_states[time_iter].mv(next_state)) \
                          + next_state.dot(self.lin_cost_states[time_iter])

        cost_ctrl = 0.5 * ctrl.dot(self.quad_cost_ctrls[time_iter-1].mv(ctrl)) \
                    + ctrl.dot(self.lin_cost_ctrls[time_iter-1])

        return cost_next_state, cost_ctrl


def compute_lin_quad_approx(traj, costs, reg_ctrl=0.):
    lin_dyn_states = [state.grad_dyn_state.t() for state in traj[1:]]
    lin_dyn_ctrls = [state.grad_dyn_ctrl.t() for state in traj[1:]]

    quad_cost_states = [cost.hess_state for cost in costs]
    lin_cost_states = [cost.grad_state for cost in costs]

    dim_ctrl = lin_dyn_ctrls[0].shape[1]
    quad_cost_ctrls = [cost.hess_ctrl + reg_ctrl * torch.eye(dim_ctrl) for cost in costs[1:]]
    lin_cost_ctrls = [cost.grad_ctrl for cost in costs[1:]]

    return lin_dyn_states, lin_dyn_ctrls,\
           quad_cost_states, lin_cost_states,\
           quad_cost_ctrls, lin_cost_ctrls


def make_synth_linear_env(horizon=30, dim_state=4, dim_ctrl=3, seed=0):
    torch.random.manual_seed(seed)
    lin_dyn_state = 0.5*torch.rand(dim_state, dim_state)
    lin_dyn_ctrl = 0.5*torch.rand(dim_state, dim_ctrl)

    quad_cost_state = torch.rand(dim_state, dim_state)
    quad_cost_state = quad_cost_state.mm(quad_cost_state.t())
    lin_cost_state = torch.rand(dim_state)

    quad_cost_ctrl = torch.rand(dim_ctrl, dim_ctrl)
    quad_cost_ctrl = quad_cost_ctrl.mm(quad_cost_ctrl.t())
    lin_cost_ctrl = torch.rand(dim_ctrl)

    lin_dyn_states = [lin_dyn_state]*horizon
    lin_dyn_ctrls = [lin_dyn_ctrl]*horizon

    quad_cost_states = [torch.zeros_like(quad_cost_state)] + [quad_cost_state] * horizon
    lin_cost_states = [torch.zeros_like(lin_cost_state)] + [lin_cost_state] * horizon
    quad_cost_ctrls = [quad_cost_ctrl] * horizon
    lin_cost_ctrls = [lin_cost_ctrl] * horizon

    state0 = torch.rand(dim_state)

    lin_quad_env = LinQuadEnv(lin_dyn_states, lin_dyn_ctrls,
                              quad_cost_states, lin_cost_states,
                              quad_cost_ctrls, lin_cost_ctrls,
                              state0)
    return lin_quad_env
