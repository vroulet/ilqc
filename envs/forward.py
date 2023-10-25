import torch
import time
from typing import Callable, List, Tuple

from torch.autograd import grad as auto_grad
from envs.torch_utils import auto_multi_grad


class DiffEnv:
    """
    Super class that takes an environment whose dynamics/costs are coded in Pytorch
    and outputs an environment that can handle second order oracles with quadratic approximations

    Note: all environments assume decomposable costs in state and control variables, i.e.,
    costs of the form h(x, u) = h(x) + h(u) for x the state and u the control
    """
    def __init__(self):
        super(DiffEnv, self).__init__()
        self.dim_ctrl, self.dim_state = None, None
        self.dt, self.horizon = None, None
        self.state, self.time_iter = None, 0
        self.init_state, self.init_time_iter = None, None
        self.viewer = None

    def discrete_dyn(self, state: torch.Tensor, ctrl: torch.Tensor, time_iter: int) -> torch.Tensor:
        """
        Core step of the environment, to be replaced by the actual physics system
        :param state: current state
        :param ctrl: applied control variable
        :param time_iter: time index
        :return: next state of the system
        """
        raise NotImplementedError

    def costs(self, next_state: torch.Tensor, ctrl: torch.Tensor, time_iter: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cost associated to the state that has just been computed and the cost associated to the applied control variable
        To be replaced by the given instance of the problem
        :param next_state: next state computed by one step of the environment
        :param ctrl: applied control variable
        :param time_iter: time index
        :return:
            - cost_next_state - cost on the next state
            - cost_ctrl - cost associated to the given control variable
        """
        return torch.tensor(0.), torch.tensor(0.)

    def step(self, ctrl: torch.Tensor, approx: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step of the environment, uses current state of the environment (treated as an attribute as in gym) and make
        one step.
        Here the first or second order information of the dynamics and the costs are potentially recorded to compute
        optimization oracles. This information are directly stored as attributes to next_state and cost

        :param ctrl: action taken at this step
        :param approx: flag to know if the intermediate first/second order information needs to be stored
                approx == 'linquad': record gradients of the dynamics, gradients and hessians of the costs
                approx == 'quad' record gradients, hessians of the dynamics, gradients and hessians of the costs
                approx == None, use the default dynamics without recording anything
                (gradients may be still be stored by pytorch to compute e.g. a gradient)
        :return:
            - next_state - next state of the system
            - cost - cost corresponding to the new state and the control used
        """
        if approx in ['linquad', 'quad']:
            next_state = diff_discrete_dyn(self.discrete_dyn, self.state, ctrl, self.time_iter,
                                           quad_approx=approx == 'quad')
            new_time_iter = self.time_iter + 1
            cost = diff_cost(self.costs, next_state, ctrl, new_time_iter)
        else:
            next_state = self.discrete_dyn(self.state, ctrl, self.time_iter)
            new_time_iter = self.time_iter + 1
            cost = sum(self.costs(next_state, ctrl, new_time_iter))
        self.state, self.time_iter = next_state, new_time_iter
        return next_state, cost

    def forward(self, cmd: torch.Tensor, approx: str = None, reset: bool = True) \
            -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute trajectories and costs associated to a sequence of control variables
        Can record first/second order information on dynamics/costs depending on the input approx (see function step)
        :param cmd: sequence of controls to be applied on the system given as a tensor of size (nb_steps, dim_ctrl)
        :param approx: what approximations of the dynamics/costs need to be stored (see function step)
        :param reset: whether to reset the initial state or use the current state of the system
        :return:
            - traj - trajectory of the system recorded as a list of states with potentially first or second
            information on the dynamics stored as attributes of the states
            - costs - costs recorded along the movement recorded as a list of costs with potentially first and second
            information on the costs stored as attributes of the states
        """
        if reset:
            self.reset(requires_grad=approx is not None)
        dim_state, dim_ctrl, horizon = self.dim_state, self.dim_ctrl, cmd.shape[0]
        state = self.state
        traj = [state]
        init_cost = torch.tensor(0.)
        init_cost.hess_state, init_cost.grad_state = torch.zeros(dim_state, dim_state), torch.zeros(dim_state)
        costs = [init_cost]
        for t in range(horizon):
            state, cost = self.step(cmd[t], approx=approx)
            traj.append(state)
            costs.append(cost)
        return traj, costs

    def reset(self, requires_grad=False) -> torch.Tensor:
        """
        Reset the environment to its initial state and its initial time index
        :param requires_grad: whether gradients associated to the state variables need to be computed
        :return:
            initial state
        """
        self.time_iter = self.init_time_iter
        self.state = self.init_state
        self.state.requires_grad = requires_grad
        return self.state

    def close(self):
        """
        Close the viewer for visualisation
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def visualize(self, cmd: torch.Tensor, dt: float =None):
        """
        Visualize the movement associated to a sequence of control variables
        :param cmd: sequence of controls to be applied on the system given as a tensor of size (nb_steps, dim_ctrl)
        :param dt: time step to visualize the movement (default is to use the time step defined in the environment)
        """
        if dt is None:
            dt = self.dt
        self.reset()
        for ctrl in cmd:
            self.render()
            time.sleep(dt)
            self.step(ctrl)
        self.render()
        self.close()

    def render(self, title=None):
        """
        Visualize the environment after the given action
        :param title:
        :return: viewer
        """
        raise NotImplementedError


def diff_discrete_dyn(discrete_dyn: Callable, state: torch.Tensor,
                      ctrl: torch.Tensor, time_iter: int, quad_approx: bool = False) -> torch.Tensor:
    """
    Make one step and record first/second order oracles
    :param discrete_dyn: discrete dynamics to compute derivatives from
    :param state: state of the environment
    :param ctrl: control variables of the environment
    :param time_iter: time index
    :param quad_approx: whether to record quadratic approximation
    :return:
    next_state: next state filled with first/second order information of the dynamic w.r.t. the ctrl and the state
    """
    next_state = discrete_dyn(state, ctrl, time_iter)

    grad_dyn_state = auto_multi_grad(next_state, state, create_graph=quad_approx)
    grad_dyn_ctrl = auto_multi_grad(next_state, ctrl, create_graph=quad_approx)
    for attr, val in zip(['grad_dyn_state', 'grad_dyn_ctrl'], [grad_dyn_state, grad_dyn_ctrl]):
        val = val.data if val is not None else None
        setattr(next_state, attr, val)
    if quad_approx:
        if grad_dyn_state.grad_fn is None:
            def hess_dyn_state(lam):
                return torch.zeros(state.shape[0], state.shape[0])

            def hess_dyn_state_ctrl(lam):
                return torch.zeros(state.shape[0], ctrl.shape[0])
        else:
            def hess_dyn_state(lam):
                return auto_multi_grad(grad_dyn_state.mv(lam), state)

            def hess_dyn_state_ctrl(lam):
                try:
                    output = auto_multi_grad(grad_dyn_state.mv(lam), ctrl).t()
                except RuntimeError as err:
                    if 'One of the differentiated Tensors appears to not have been used in the graph' in str(err):
                        output = torch.zeros(state.shape[0], ctrl.shape[0])
                    else:
                        raise err
                return output

        if grad_dyn_ctrl.grad_fn is None:
            def hess_dyn_ctrl(lam):
                return torch.zeros(ctrl.shape[0], ctrl.shape[0])
        else:
            def hess_dyn_ctrl(lam):
                try:
                    output = auto_multi_grad(grad_dyn_ctrl.mv(lam), ctrl)
                except RuntimeError as err:
                    if 'One of the differentiated Tensors appears to not have been used in the graph' in str(err):
                        output = torch.zeros(ctrl.shape[0], ctrl.shape[0])
                    else:
                        raise err
                return output

        for attr, val in zip(['hess_dyn_state', 'hess_dyn_ctrl', 'hess_dyn_state_ctrl'],
                             [hess_dyn_state, hess_dyn_ctrl, hess_dyn_state_ctrl]):
            setattr(next_state, attr, val)

    return next_state


def diff_cost(costs: Callable, next_state: torch.Tensor, ctrl: torch.Tensor, time_iter: int) -> torch.Tensor:
    """
    Compute costs associated to the new state and the control applied and record linear quadratic approximation of
    the costs
    :param costs: pre-defined costs for the given environment
    :param next_state: state that has been computed by oen step of the dynamic
    :param ctrl: applied control variable
    :param time_iter: time index
    :return: cost associated to the new state and the control applied to the system, gradients and hessians of the
            costs are recorded as attributes to the variable cost.
    """
    dim_state, dim_ctrl = next_state.shape[0], ctrl.shape[0]

    cost_next_state, cost_ctrl = costs(next_state, ctrl, time_iter)
    cost = cost_next_state + cost_ctrl

    if cost_next_state.grad_fn is not None:
        cost_grad_state = auto_grad(cost_next_state, next_state, create_graph=True)[0]
        cost.hess_state = auto_multi_grad(cost_grad_state, next_state).data
        cost.grad_state = cost_grad_state.data
        cost.has_cost_state = True
    else:
        cost.hess_state, cost.grad_state = torch.zeros(dim_state, dim_state), torch.zeros(dim_state)
        cost.has_cost_state = False

    if cost_ctrl.grad_fn is not None:
        cost_grad_ctrl = auto_grad(cost_ctrl, ctrl, create_graph=True)[0]
        cost.hess_ctrl = auto_multi_grad(cost_grad_ctrl, ctrl).data
        cost.grad_ctrl = cost_grad_ctrl.data
    else:
        cost.hess_ctrl, cost.grad_ctrl = torch.zeros(dim_ctrl, dim_ctrl), torch.zeros(dim_ctrl)

    return cost



